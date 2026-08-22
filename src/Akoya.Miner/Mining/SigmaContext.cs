using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using Akoya.Crypto;
using Akoya.Miner.Stratum;
using PearlPool.Proto.V2;

namespace Akoya.Miner.Mining;

internal sealed class SigmaContext
{
    public const int HeaderSize = 76;
    public const int ConfigSize = 52;
    public const int BSeedSize = 32;
    public const uint CertVersionV2 = 2;
    public const uint CertVersionV3 = 3;
    public const ulong LuckyPoolMainnetSaltedSeedForkHeight = 98_900;

    public Guid JobId { get; }

    /// <summary>
    /// External pool job identifier.
    ///
    /// Akoya V2 normally uses a 16-byte UUID internally, while Stratum pools
    /// such as LuckyPool use arbitrary string identifiers such as:
    ///
    ///     e6b8e647_500000
    ///
    /// For Akoya V2 jobs this is null.
    /// </summary>
    public string? ExternalJobId { get; }

    public byte[] Sigma { get; }                    // 76 B incomplete header
    public byte[] ConfigBytes { get; }              // 52 B MiningConfiguration.ToBytes()
    public uint CommonDim { get; }                  // = K (miner-chosen)
    public ushort Rank { get; }                     // = R (miner-chosen, default 128)
    public uint CertVersion { get; }
    public bool UseSaltedSeeds => CertVersion == CertVersionV3;
    public byte[] JobKey { get; }                   // 32 B BLAKE3 keyed merkle key
    public byte[] BSeed { get; }

    /// <summary>
    /// Audit-proof v1 K parameter (0 disabled, ≤64 per spec).
    ///
    /// When >0 every ShareSubmission carries a K-opening AuditProof keyed by
    /// (claimed_hash, b_seed, K) per the audit_proof v1 schematic.
    ///
    /// LuckyPool Stratum mode uses AuditK=0 because the Akoya-specific
    /// pool audit protocol does not exist there.
    /// </summary>
    public uint AuditK { get; }

    /// <summary>
    /// Akoya V2 compact share target.
    ///
    /// In LuckyPool mode this is zero and ExplicitTarget contains the actual
    /// full 256-bit target supplied by the Stratum pool.
    /// </summary>
    public uint TargetNbits { get; }

    public uint NetworkTargetNbits { get; }

    /// <summary>
    /// Full target supplied directly by an external pool.
    ///
    /// Null for normal Akoya V2 jobs.
    /// Non-null for LuckyPool / standard Pearl Stratum jobs.
    /// </summary>
    public BigInteger? ExplicitTarget { get; }

    /// <summary>
    /// Target that the mining pipeline should actually use.
    ///
    /// Akoya V2:
    ///     TargetNbits -> NbitsToTarget()
    ///
    /// LuckyPool:
    ///     the exact 256-bit target received from mining.notify
    /// </summary>
    public BigInteger EffectiveTarget =>
        ExplicitTarget ?? NbitsToTarget(TargetNbits);

    public long BlockHeight { get; }

    private SigmaContext(
        Guid jobId,
        byte[] sigma,
        byte[] configBytes,
        uint commonDim,
        ushort rank,
        uint certVersion,
        byte[] jobKey,
        byte[] bSeed,
        uint auditK,
        uint targetNbits,
        uint networkTargetNbits,
        long blockHeight,
        BigInteger? explicitTarget = null,
        string? externalJobId = null)
    {
        JobId              = jobId;
        ExternalJobId      = externalJobId;
        Sigma              = sigma;
        ConfigBytes        = configBytes;
        CommonDim          = commonDim;
        Rank               = rank;
        CertVersion        = certVersion;
        JobKey             = jobKey;
        BSeed              = bSeed;
        AuditK             = auditK;
        TargetNbits        = targetNbits;
        NetworkTargetNbits = networkTargetNbits;
        ExplicitTarget     = explicitTarget;
        BlockHeight        = blockHeight;
    }

    /// <summary>
    /// True iff <paramref name="job"/> carries the structural minimum required
    /// to build a <see cref="SigmaContext"/> — i.e. a 16-byte UUID job_id and a
    /// header-sized sigma.
    ///
    /// Used as a pre-publish guard on paths where the pool may return a
    /// "session resumed but no current job yet" response
    /// (Resume Success=true with empty job_id/sigma): the orchestrator must
    /// accept the session without crashing and wait for the next OnJob over
    /// the bidi stream, rather than blow up and trip reconnect on a perfectly
    /// authenticated session.
    ///
    /// Pool integration doc §2.4 commits Resume to always carry the current job
    /// if one exists; if pool returns success without it, treat it as
    /// "no current job — stream will deliver".
    /// </summary>
    public static bool IsValidInitialJob(JobAssignment job)
    {
        if (job is null)
            return false;

        if (job.JobId is null || job.JobId.Length != 16)
            return false;

        if (job.Sigma is null || job.Sigma.Length != HeaderSize)
            return false;

        // audit_proof v1 requires a 32 B b_seed.
        //
        // If the pool's ResumeResponse omitted it (e.g. an older pool
        // version), fall through to the MiningStream OnJob path instead of
        // crashing the orchestrator.
        if (job.BSeed is null || job.BSeed.Length != BSeedSize)
            return false;

        return true;
    }

    /// <summary>
    /// Parse an Akoya V2 JobAssignment into the immutable per-σ snapshot.
    /// </summary>
    /// <param name="job">The JobAssignment the pool just pushed.</param>
    /// <param name="minerId">
    /// 16-byte minerId assigned by Register (little-endian Guid byte layout —
    /// i.e. what arrives on the wire as RegisterResponse.miner_id).
    /// </param>
    /// <param name="commonDim">Miner's chosen K.</param>
    /// <param name="rank">Miner's chosen R (default 128).</param>
    public static SigmaContext FromJobAssignment(
        JobAssignment job,
        ReadOnlySpan<byte> minerId,
        uint commonDim,
        ushort rank)
    {
        if (job.Sigma.Length != HeaderSize)
        {
            throw new InvalidOperationException(
                $"JobAssignment.sigma length {job.Sigma.Length} != expected " +
                $"{HeaderSize} (V2: header-only)");
        }

        if (minerId.Length != 16)
        {
            throw new ArgumentException(
                "minerId must be 16 B",
                nameof(minerId));
        }

        if (job.JobId.Length != 16)
        {
            throw new InvalidOperationException(
                "JobAssignment.job_id must be 16 B");
        }

        if (commonDim == 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(commonDim),
                "commonDim (K) must be non-zero");
        }

        if (rank == 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(rank),
                "rank (R) must be non-zero");
        }

        if (job.BSeed.Length != BSeedSize)
        {
            throw new InvalidOperationException(
                $"JobAssignment.b_seed length {job.BSeed.Length} != expected " +
                $"{BSeedSize} (audit_proof v1)");
        }

        if (job.AuditK > AuditIndexDeriver.AuditKMax)
        {
            throw new InvalidOperationException(
                $"JobAssignment.audit_k {job.AuditK} > spec cap " +
                $"{AuditIndexDeriver.AuditKMax}");
        }

        var sigma =
            job.Sigma.ToByteArray();

        var config =
            MiningConfiguration.Default(
                commonDim,
                rank);

        var configBytes =
            config.ToBytes();

        _ = minerId;

#pragma warning disable CS0618 // 2-arg overload is canonical for V2.
        var jobKey =
            CommitmentHasher.GetKey(
                sigma,
                config);
#pragma warning restore CS0618

        return new SigmaContext(
            jobId:              new Guid(job.JobId.Span),
            sigma:              sigma,
            configBytes:        configBytes,
            commonDim:          commonDim,
            rank:               rank,
            certVersion:        CertVersionV2,
            jobKey:             jobKey,
            bSeed:              job.BSeed.ToByteArray(),
            auditK:             job.AuditK,
            targetNbits:        job.TargetNbits,
            networkTargetNbits: job.NetworkTargetNbits,
            blockHeight:        job.BlockHeight,
            explicitTarget:     null,
            externalJobId:      null);
    }

    /// <summary>
    /// Convert a LuckyPool / Pearl Stratum job into the same immutable
    /// SigmaContext consumed by the existing Akoya GPU pipeline.
    ///
    /// Important differences from Akoya V2:
    ///
    ///  - LuckyPool already provides a complete 256-bit target.
    ///  - LuckyPool does not provide an Akoya V2 b_seed.
    ///  - LuckyPool does not use Akoya's audit_proof mechanism.
    ///  - LuckyPool job_id is an arbitrary string rather than a UUID.
    ///
    /// The B seed is therefore deterministically derived locally from the
    /// actual Pearl header + mining configuration.
    /// </summary>
    public static SigmaContext FromLuckyPoolJob(
        LuckyPoolJob job,
        uint commonDim,
        ushort rank)
    {
        ArgumentNullException.ThrowIfNull(job);

        if (job.HeaderBytes.Length != HeaderSize)
        {
            throw new InvalidOperationException(
                $"LuckyPool header length {job.HeaderBytes.Length} != expected " +
                $"{HeaderSize}");
        }

        if (job.TargetBytes.Length != 32)
        {
            throw new InvalidOperationException(
                $"LuckyPool target length {job.TargetBytes.Length} != expected 32");
        }

        if (string.IsNullOrWhiteSpace(job.JobId))
        {
            throw new InvalidOperationException(
                "LuckyPool job_id must not be empty");
        }

        if (commonDim == 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(commonDim),
                "commonDim (K) must be non-zero");
        }

        if (rank == 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(rank),
                "rank (R) must be non-zero");
        }

        // ------------------------------------------------------------
        // Pearl sigma / header
        // ------------------------------------------------------------

        var sigma =
            (byte[])job.HeaderBytes.Clone();

        // ------------------------------------------------------------
        // Mining configuration
        // ------------------------------------------------------------

        var config =
            MiningConfiguration.Default(
                commonDim,
                rank);

        var configBytes =
            config.ToBytes();

        if (configBytes.Length != ConfigSize)
        {
            throw new InvalidOperationException(
                $"MiningConfiguration serialized length {configBytes.Length} " +
                $"!= expected {ConfigSize}");
        }

        // ------------------------------------------------------------
        // JobKey
        //
        // This is intentionally identical to the Akoya V2 path.
        //
        // The BSeed is NOT part of the job key.
        // ------------------------------------------------------------

#pragma warning disable CS0618
        var jobKey =
            CommitmentHasher.GetKey(
                sigma,
                config);
#pragma warning restore CS0618

        // ------------------------------------------------------------
        // Deterministic local BSeed
        //
        // Akoya V2 receives this from its pool.
        // Standard Pearl Stratum does not.
        //
        // Derive it deterministically from:
        //
        //   domain || sigma || MiningConfiguration
        //
        // This guarantees that every component of this miner regenerates
        // exactly the same B matrix for a given Stratum job/config.
        // ------------------------------------------------------------

        var domain =
            Encoding.ASCII.GetBytes(
                "LuckyPool-B-v1");

        var bSeedInput =
            new byte[
                domain.Length +
                sigma.Length +
                configBytes.Length];

        var offset = 0;

        Buffer.BlockCopy(
            domain,
            0,
            bSeedInput,
            offset,
            domain.Length);

        offset +=
            domain.Length;

        Buffer.BlockCopy(
            sigma,
            0,
            bSeedInput,
            offset,
            sigma.Length);

        offset +=
            sigma.Length;

        Buffer.BlockCopy(
            configBytes,
            0,
            bSeedInput,
            offset,
            configBytes.Length);

        var bSeed =
            Blake3.Hash(
                bSeedInput);

        if (bSeed.Length != BSeedSize)
        {
            throw new InvalidOperationException(
                $"Derived BSeed length {bSeed.Length} != expected {BSeedSize}");
        }

        // ------------------------------------------------------------
        // LuckyPool external job ID -> deterministic internal Guid
        //
        // The existing Akoya pipeline expects Guid JobId.
        //
        // Keep the original LuckyPool string separately in ExternalJobId
        // because that exact value must eventually be sent back in
        // mining.submit.
        // ------------------------------------------------------------

        Span<byte> jobIdHash =
            stackalloc byte[32];

        SHA256.HashData(
            Encoding.UTF8.GetBytes(
                job.JobId),
            jobIdHash);

        var internalJobId =
            new Guid(
                jobIdHash[..16]);

        // ------------------------------------------------------------
        // Exact LuckyPool share target
        //
        // TargetBytes are network-order / big-endian.
        // Do NOT round-trip through NBits because compact targets lose
        // precision.
        // ------------------------------------------------------------

        var explicitTarget =
            new BigInteger(
                job.TargetBytes,
                isUnsigned: true,
                isBigEndian: true);

        if (explicitTarget <= BigInteger.Zero)
        {
            throw new InvalidOperationException(
                "LuckyPool target must be positive");
        }

        // ------------------------------------------------------------
        // Height
        // ------------------------------------------------------------

        var blockHeight =
            job.Height.HasValue
                ? checked((long)job.Height.Value)
                : 0L;

        var certVersion = ResolveLuckyPoolCertVersion(job);

        return new SigmaContext(
            jobId:              internalJobId,
            sigma:              sigma,
            configBytes:        configBytes,
            commonDim:          commonDim,
            rank:               rank,
            certVersion:        certVersion,
            jobKey:             jobKey,
            bSeed:              bSeed,

            // Akoya pool-specific audit is disabled for Stratum.
            auditK:             0,

            // LuckyPool uses the full explicit target below.
            targetNbits:        0,
            networkTargetNbits: 0,

            blockHeight:        blockHeight,
            explicitTarget:     explicitTarget,
            externalJobId:      job.JobId);
    }

    /// <summary>
    /// Return a copy of this context with the Akoya share-difficulty NBits
    /// replaced.
    ///
    /// This remains the Akoya V2 vardiff path. Supplying NBits intentionally
    /// clears ExplicitTarget because the new compact target becomes
    /// authoritative.
    /// </summary>
    public SigmaContext WithTargetNbits(
        uint newTargetNbits) => new(
            jobId:              JobId,
            sigma:              Sigma,
            configBytes:        ConfigBytes,
            commonDim:          CommonDim,
            rank:               Rank,
            certVersion:        CertVersion,
            jobKey:             JobKey,
            bSeed:              BSeed,
            auditK:             AuditK,
            targetNbits:        newTargetNbits,
            networkTargetNbits: NetworkTargetNbits,
            blockHeight:        BlockHeight,
            explicitTarget:     null,
            externalJobId:      ExternalJobId);

    /// <summary>
    /// Return a copy with an exact full-width target.
    ///
    /// This will be useful for Stratum vardiff / new target updates later.
    /// </summary>
    public SigmaContext WithExplicitTarget(
        BigInteger newTarget)
    {
        if (newTarget <= BigInteger.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(newTarget),
                "Target must be positive");
        }

        return new SigmaContext(
            jobId:              JobId,
            sigma:              Sigma,
            configBytes:        ConfigBytes,
            commonDim:          CommonDim,
            rank:               Rank,
            certVersion:        CertVersion,
            jobKey:             JobKey,
            bSeed:              BSeed,
            auditK:             AuditK,
            targetNbits:        0,
            networkTargetNbits: NetworkTargetNbits,
            blockHeight:        BlockHeight,
            explicitTarget:     newTarget,
            externalJobId:      ExternalJobId);
    }

    private static uint ResolveLuckyPoolCertVersion(LuckyPoolJob job)
    {
        if (job.CertVersion.HasValue)
        {
            uint version = job.CertVersion.Value;
            if (version == 1 || version == CertVersionV2 || version == CertVersionV3)
                return version;

            throw new InvalidOperationException(
                $"LuckyPool requires unsupported Pearl certificate version {version}; refusing to guess seed rules.");
        }

        if (!job.Height.HasValue)
        {
            throw new InvalidOperationException(
                "LuckyPool job supplied neither a certificate version nor a block height; cannot select Pearl seed derivation safely.");
        }

        // Compatibility fallback for LuckyPool's historical Stratum notify,
        // which supplied height but no requiredcertversion. This adapter is
        // for Pearl mainnet; V3 activates at height 98900.
        return job.Height.Value >= LuckyPoolMainnetSaltedSeedForkHeight
            ? CertVersionV3
            : CertVersionV2;
    }

    /// <summary>
    /// Convert compact NBits target representation to a full 256-bit target.
    /// </summary>
    public static BigInteger NbitsToTarget(
        uint nbits)
    {
        int exp =
            (int)(nbits >> 24);

        uint mantissa =
            nbits & 0x00FFFFFFu;

        if (exp <= 3)
        {
            return new BigInteger(
                mantissa >> (8 * (3 - exp)));
        }

        return
            new BigInteger(mantissa)
            << (8 * (exp - 3));
    }

    /// <summary>
    /// SHA-256(σ) first 8 B — mirrors the server's sigma fingerprint
    /// so we can compare quickly across heartbeats without round-tripping σ.
    /// </summary>
    public static byte[] Fingerprint(
        ReadOnlySpan<byte> sigma)
    {
        if (sigma.IsEmpty)
            return [];

        Span<byte> full =
            stackalloc byte[32];

        SHA256.HashData(
            sigma,
            full);

        var fp =
            new byte[8];

        full[..8].CopyTo(
            fp);

        return fp;
    }
}
