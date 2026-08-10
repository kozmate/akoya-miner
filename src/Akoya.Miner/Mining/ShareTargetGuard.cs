using System.Numerics;
using Akoya.Crypto;
using Akoya.Mining;

namespace Akoya.Miner.Mining;

/// <summary>
/// Final CPU-side target check for a fully built candidate share.
///
/// The GPU performs the high-throughput search and only signals candidates
/// whose device-side jackpot hash appears to clear the installed target.
/// This guard independently compares the CPU-built share's ClaimedHash against
/// the exact same adjusted target before handing it to the share sink.
/// </summary>
internal static class ShareTargetGuard
{
    /// <summary>
    /// Backward-compatible Akoya V2 path using compact NBits.
    /// </summary>
    public static bool ClearsLiveTarget(
        ReadOnlySpan<byte> claimedHashLittleEndian,
        uint installedNbits,
        MiningConfiguration cfg)
    {
        return ClearsLiveTarget(
            claimedHashLittleEndian,
            SigmaContext.NbitsToTarget(installedNbits),
            cfg);
    }

    /// <summary>
    /// Unified path using the exact 256-bit pool target that was installed on
    /// the GPU for this candidate. The protocol DifficultyAdjustmentFactor is
    /// applied here exactly as it is on the device-side PowTarget.
    /// </summary>
    public static bool ClearsLiveTarget(
        ReadOnlySpan<byte> claimedHashLittleEndian,
        BigInteger installedPoolTarget,
        MiningConfiguration cfg)
    {
        if (installedPoolTarget <= BigInteger.Zero)
            return false;

        var adjustedTarget = installedPoolTarget * cfg.DifficultyAdjustmentFactor();
        var maxTarget = (BigInteger.One << 256) - BigInteger.One;
        if (adjustedTarget > maxTarget)
            adjustedTarget = maxTarget;

        var hashInt = new BigInteger(
            claimedHashLittleEndian,
            isUnsigned: true,
            isBigEndian: false);

        return hashInt <= adjustedTarget;
    }
}
