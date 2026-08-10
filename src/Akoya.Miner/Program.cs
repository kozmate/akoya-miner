// Akoya.Miner v2
//
// Subcommands:
//   mine-blocks               Connect to Akoya V2 pool, register/resume, mine.
//   luckypool-test            Connect/auth to LuckyPool, print first real job, exit.
//   luckypool-mine-test       Experimental LuckyPool GPU dry-run; NEVER submits shares.
//   luckypool-submit-test     Submit exactly one locally validated LuckyPool share, print result, exit.
//   luckypool-mine            Continuous LuckyPool Stratum mining with live share submission + reconnect.
//   perf-iter                 Profile the steady-state GPU iteration loop (no pool/network required).
//   version | --version | -V  Print git sha + miner version.
//
// Runtime native libs:
//   AKOYA_PEARL_GEMM_LIB    absolute path to libpearl_gemm_capi.so
//   AKOYA_PEARL_MINING_LIB  absolute path to libpearl_mining_capi.so
//   (Unset → falls through to the OS loader via LD_LIBRARY_PATH.)
//
// All other configuration is read once at startup by EnvVarBindings.Load.

using System.Diagnostics;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Akoya.Miner.Config;
using Akoya.Miner.Mining;
using Akoya.Miner.Stratum;
using Akoya.Cuda;
using Akoya.Miner.Observability;
using Akoya.Mining;
using Akoya.PearlGemm;
using Akoya.Pool;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Logging.Console;

// On WSL the kernel-side libcuda lives at /usr/lib/wsl/lib/libcuda.so.1 and the
// stale dpkg-installed libcuda in /usr/lib/x86_64-linux-gnu wins under ldconfig.
// Loading the latter inside WSL returns CUDA_ERROR_NO_DEVICE (100) from cuInit.
// Prefer the WSL stub when it exists. Same logic the test module-initializer
// uses; mirrored here so production miners on WSL don't fail to enumerate GPUs.
NativeLibrary.SetDllImportResolver(typeof(CudaDriver).Assembly, (name, _, _) =>
{
    if (name != "cuda") return 0;
    // Windows: the CUDA driver API lives in nvcuda.dll (installed with the
    // GPU driver), not libcuda.so. None of the WSL/ROCm .so logic applies.
    if (OperatingSystem.IsWindows())
        return NativeLibrary.Load("nvcuda.dll");
    const string wslLibCuda = "/usr/lib/wsl/lib/libcuda.so.1";
    if (OperatingSystem.IsLinux() && File.Exists(wslLibCuda))
    {
        try { return NativeLibrary.Load(wslLibCuda); }
        catch { /* fall through to default */ }
    }
    // The ROCm backend stages a libcuda.so.1 shim next to the binary.
    var localCuda = Path.Combine(AppContext.BaseDirectory, "libcuda.so.1");
    if (File.Exists(localCuda))
    {
        try { return NativeLibrary.Load(localCuda); }
        catch { /* fall through to default */ }
    }
    return NativeLibrary.Load("libcuda.so.1");
});

NativeLibrary.SetDllImportResolver(typeof(PearlGemmNative).Assembly, (name, _, _) =>
    name == PearlGemmNative.Lib
        ? NativeLibs.Load("AKOYA_PEARL_GEMM_LIB", NativeLibs.GemmFile)
        : 0);

NativeLibrary.SetDllImportResolver(typeof(PearlMiningNative).Assembly, (name, _, _) =>
    name == PearlMiningNative.Lib
        ? NativeLibs.Load("AKOYA_PEARL_MINING_LIB", NativeLibs.MiningFile)
        : 0);

// Last-resort crash recorder. The fleet runs without easy log retrieval and
// .NET's createdump needs DOTNET_DbgEnableMiniDump=1 set BEFORE managed code
// starts (we can only warn about it here, not set it). This handler at least
// writes a structured plain-text record on any unhandled exception so an
// operator can mail us a single file. Best-effort only; never throws.
AppDomain.CurrentDomain.UnhandledException += (_, ev) =>
{
    try
    {
        var dir = CrashDumpHelpers.ResolveDumpDir();
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "last-fatal.log");
        var sb = new StringBuilder();
        sb.Append("ts=").Append(DateTime.UtcNow.ToString("o")).AppendLine();
        sb.Append("miner_version=").Append(VersionInfo.MinerVersion).AppendLine();
        sb.Append("git_sha=").Append(VersionInfo.GitSha).AppendLine();
        sb.Append("terminating=").Append(ev.IsTerminating).AppendLine();
        sb.AppendLine("---");
        sb.AppendLine(ev.ExceptionObject?.ToString() ?? "(no exception object)");
        File.WriteAllText(path, sb.ToString());
    }
    catch { /* swallow — handler must never throw */ }
};

var cmd = args.Length > 0 ? args[0] : "mine-blocks";
return cmd switch
{
    "mine-blocks"                    => await MineBlocksAsync(args),
    "luckypool-test"                 => await LuckyPoolTestAsync(args),
    "luckypool-mine-test"            => await LuckyPoolMineTestAsync(args),
    "luckypool-submit-test"          => await LuckyPoolSubmitTestAsync(args),
    "luckypool-mine"                 => await LuckyPoolMineAsync(args),
    "perf-iter"                      => PerfIter(args),
    "selftest" or "--selftest"       => await SelfTestAsync(args),
    "version" or "--version" or "-V" => PrintVersion(),
    _                                => Usage(cmd),
};

static async Task<int> LuckyPoolTestAsync(string[] _)
{
    var host =
        Environment.GetEnvironmentVariable("LUCKYPOOL_HOST")
        ?? "pearl-eu2.luckypool.io";

    var portString =
        Environment.GetEnvironmentVariable("LUCKYPOOL_PORT")
        ?? "3360";

    var wallet =
        Environment.GetEnvironmentVariable("LUCKYPOOL_WALLET");

    if (string.IsNullOrWhiteSpace(wallet))
    {
        Console.Error.WriteLine(
            "ERROR: LUCKYPOOL_WALLET environment variable is required.");

        return 78;
    }

    var worker =
        Environment.GetEnvironmentVariable("LUCKYPOOL_WORKER")
        ?? Environment.MachineName;

    if (!int.TryParse(portString, out var port) ||
        port < 1 ||
        port > 65535)
    {
        Console.Error.WriteLine(
            $"ERROR: Invalid LUCKYPOOL_PORT: {portString}");

        return 78;
    }

    using var cts = new CancellationTokenSource();

    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;

        Console.WriteLine();
        Console.WriteLine(
            "LuckyPool: Ctrl-C received, shutting down...");

        cts.Cancel();
    };

    try
    {
        await using var client =
            new LuckyPoolClient(
                host,
                port,
                wallet,
                worker);

        await client.RunUntilFirstJobAsync(
            cts.Token);

        return 0;
    }
    catch (OperationCanceledException)
    {
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine();
        Console.Error.WriteLine(
            "LuckyPool test failed:");
        Console.Error.WriteLine(ex);

        return 1;
    }
}


static async Task<int> LuckyPoolMineTestAsync(string[] _)
{
    using var loggerFactory = BuildLoggerFactory();
    var log = loggerFactory.CreateLogger("luckypool-mine-test");

    var host =
        Environment.GetEnvironmentVariable("LUCKYPOOL_HOST")
        ?? "pearl-eu2.luckypool.io";

    var portString =
        Environment.GetEnvironmentVariable("LUCKYPOOL_PORT")
        ?? "3360";

    var wallet =
        Environment.GetEnvironmentVariable("LUCKYPOOL_WALLET");

    if (string.IsNullOrWhiteSpace(wallet))
    {
        log.LogError("LUCKYPOOL_WALLET environment variable is required.");
        return 78;
    }

    var worker =
        Environment.GetEnvironmentVariable("LUCKYPOOL_WORKER")
        ?? Environment.MachineName;

    if (!int.TryParse(portString, out var port) || port < 1 || port > 65535)
    {
        log.LogError("Invalid LUCKYPOOL_PORT: {Port}", portString);
        return 78;
    }

    // EnvVarBindings still owns the common GPU/mining configuration.  It also
    // expects the normal Akoya pool identity variables, even though this
    // subcommand never opens an Akoya gRPC connection. Mirror the LuckyPool
    // identity only when the caller did not explicitly set the Akoya values.
    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_POOL_WALLET")))
        Environment.SetEnvironmentVariable("AKOYA_POOL_WALLET", wallet);
    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_POOL_WORKER")))
        Environment.SetEnvironmentVariable("AKOYA_POOL_WORKER", worker);

    MinerOptions opts;
    try
    {
        opts = EnvVarBindings.Load(log);
    }
    catch (Exception ex)
    {
        log.LogError(ex, "LuckyPool dry-run: configuration error");
        return 78;
    }

    log.LogWarning(
        "LuckyPool GPU DRY-RUN — host={Host}:{Port} wallet={Wallet} worker={Worker} GPUs={Gpus}; finalized candidates are logged and DROPPED, never submitted",
        host, port, wallet, worker, opts.Gpus.IndicesRaw);

    using var cts = new CancellationTokenSource();

    static void TryCancel(CancellationTokenSource c)
    {
        try { c.Cancel(); }
        catch (ObjectDisposedException) { /* shutdown race — fine */ }
    }

    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        log.LogInformation("LuckyPool dry-run: Ctrl-C received — stopping workers");
        TryCancel(cts);
    };

    using var sigTerm = PosixSignalRegistration.Create(PosixSignal.SIGTERM, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("LuckyPool dry-run: SIGTERM received — stopping workers");
        TryCancel(cts);
    });

    using var sigHup = PosixSignalRegistration.Create(PosixSignal.SIGHUP, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("LuckyPool dry-run: SIGHUP received — stopping workers");
        TryCancel(cts);
    });

    AppDomain.CurrentDomain.ProcessExit += (_, _) => TryCancel(cts);

    using var shutdownDeadline = ShutdownDeadline.Arm(
        cts.Token,
        TimeSpan.FromSeconds(30),
        () => Environment.Exit(ShutdownDeadline.HardExitCode),
        log);

    try
    {
        await using var client = new LuckyPoolClient(host, port, wallet, worker);
        await using var orchestrator = new WorkerOrchestrator(opts, loggerFactory);

        await orchestrator
            .RunLuckyPoolDryRunAsync(client, cts.Token)
            .ConfigureAwait(false);

        return 0;
    }
    catch (OperationCanceledException) when (cts.IsCancellationRequested)
    {
        return 0;
    }
    catch (Exception ex)
    {
        log.LogError(ex, "LuckyPool GPU dry-run failed");
        return 1;
    }
}

static async Task<int> LuckyPoolSubmitTestAsync(string[] _)
{
    using var loggerFactory = BuildLoggerFactory();
    var log = loggerFactory.CreateLogger("luckypool-submit-test");

    var host =
        Environment.GetEnvironmentVariable("LUCKYPOOL_HOST")
        ?? "pearl-eu2.luckypool.io";

    var portString =
        Environment.GetEnvironmentVariable("LUCKYPOOL_PORT")
        ?? "3360";

    var wallet =
        Environment.GetEnvironmentVariable("LUCKYPOOL_WALLET");

    if (string.IsNullOrWhiteSpace(wallet))
    {
        log.LogError("LUCKYPOOL_WALLET environment variable is required.");
        return 78;
    }

    var worker =
        Environment.GetEnvironmentVariable("LUCKYPOOL_WORKER")
        ?? Environment.MachineName;

    if (!int.TryParse(portString, out var port) || port < 1 || port > 65535)
    {
        log.LogError("Invalid LUCKYPOOL_PORT: {Port}", portString);
        return 78;
    }

    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_POOL_WALLET")))
        Environment.SetEnvironmentVariable("AKOYA_POOL_WALLET", wallet);
    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_POOL_WORKER")))
        Environment.SetEnvironmentVariable("AKOYA_POOL_WORKER", worker);

    MinerOptions opts;
    try
    {
        opts = EnvVarBindings.Load(log);
    }
    catch (Exception ex)
    {
        log.LogError(ex, "LuckyPool submit-test: configuration error");
        return 78;
    }

    log.LogWarning(
        "LuckyPool LIVE ONE-SHARE SUBMIT TEST — host={Host}:{Port} wallet={Wallet} worker={Worker} GPUs={Gpus}. The first locally validated current-job PlainProof WILL be submitted; the process exits after the pool response.",
        host, port, wallet, worker, opts.Gpus.IndicesRaw);

    using var cts = new CancellationTokenSource();

    static void TryCancel(CancellationTokenSource c)
    {
        try { c.Cancel(); }
        catch (ObjectDisposedException) { /* shutdown race — fine */ }
    }

    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        log.LogInformation("LuckyPool submit-test: Ctrl-C received — stopping workers");
        TryCancel(cts);
    };

    using var sigTerm = PosixSignalRegistration.Create(PosixSignal.SIGTERM, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("LuckyPool submit-test: SIGTERM received — stopping workers");
        TryCancel(cts);
    });

    using var sigHup = PosixSignalRegistration.Create(PosixSignal.SIGHUP, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("LuckyPool submit-test: SIGHUP received — stopping workers");
        TryCancel(cts);
    });

    AppDomain.CurrentDomain.ProcessExit += (_, _) => TryCancel(cts);

    using var shutdownDeadline = ShutdownDeadline.Arm(
        cts.Token,
        TimeSpan.FromSeconds(30),
        () => Environment.Exit(ShutdownDeadline.HardExitCode),
        log);

    try
    {
        await using var client = new LuckyPoolClient(host, port, wallet, worker);
        await using var orchestrator = new WorkerOrchestrator(opts, loggerFactory);

        var result = await orchestrator
            .RunLuckyPoolSubmitTestAsync(client, cts.Token)
            .ConfigureAwait(false);

        if (result.Accepted)
        {
            log.LogWarning(
                "LuckyPool submit-test COMPLETE: ACCEPTED requestId={Id} result={Result}",
                result.RequestId,
                result.ResultJson);
            return 0;
        }

        log.LogWarning(
            "LuckyPool submit-test COMPLETE: REJECTED requestId={Id} result={Result} error={Error}",
            result.RequestId,
            result.ResultJson,
            result.ErrorJson ?? "(none)");
        return 2;
    }
    catch (OperationCanceledException) when (cts.IsCancellationRequested)
    {
        log.LogInformation("LuckyPool submit-test cancelled before a pool response was received.");
        return 130;
    }
    catch (Exception ex)
    {
        log.LogError(ex, "LuckyPool one-share submit-test failed");
        return 1;
    }
}

static async Task<int> LuckyPoolMineAsync(string[] _)
{
    using var loggerFactory = BuildLoggerFactory();
    var log = loggerFactory.CreateLogger("luckypool-mine");

    var host =
        Environment.GetEnvironmentVariable("LUCKYPOOL_HOST")
        ?? "pearl-eu2.luckypool.io";

    var portString =
        Environment.GetEnvironmentVariable("LUCKYPOOL_PORT")
        ?? "3360";

    var wallet =
        Environment.GetEnvironmentVariable("LUCKYPOOL_WALLET");

    if (string.IsNullOrWhiteSpace(wallet))
    {
        log.LogError("LUCKYPOOL_WALLET environment variable is required.");
        return 78;
    }

    var worker =
        Environment.GetEnvironmentVariable("LUCKYPOOL_WORKER")
        ?? Environment.MachineName;

    if (!int.TryParse(portString, out var port) || port < 1 || port > 65535)
    {
        log.LogError("Invalid LUCKYPOOL_PORT: {Port}", portString);
        return 78;
    }

    // EnvVarBindings owns the common GPU/mining settings and still validates
    // the normal Akoya pool identity fields. Mirror the LuckyPool identity
    // only when those compatibility variables were not explicitly provided.
    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_POOL_WALLET")))
        Environment.SetEnvironmentVariable("AKOYA_POOL_WALLET", wallet);
    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_POOL_WORKER")))
        Environment.SetEnvironmentVariable("AKOYA_POOL_WORKER", worker);

    MinerOptions opts;
    try
    {
        opts = EnvVarBindings.Load(log);
    }
    catch (Exception ex)
    {
        log.LogError(ex, "LuckyPool mining: configuration error");
        return 78;
    }

    if (opts.Mine.NoiseRank != 128)
    {
        log.LogError(
            "LuckyPool mining requires noise rank 128; configured R={Rank}.",
            opts.Mine.NoiseRank);
        return 78;
    }

    log.LogWarning(
        "LuckyPool LIVE MINING — host={Host}:{Port} wallet={Wallet} worker={Worker} GPUs={Gpus}; locally validated PlainProof shares WILL be submitted continuously",
        host,
        port,
        wallet,
        worker,
        opts.Gpus.IndicesRaw);

    using var cts = new CancellationTokenSource();

    static void TryCancel(CancellationTokenSource c)
    {
        try { c.Cancel(); }
        catch (ObjectDisposedException) { /* shutdown race */ }
    }

    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        log.LogInformation("LuckyPool mining: Ctrl-C received — graceful shutdown");
        TryCancel(cts);
    };

    using var sigTerm = PosixSignalRegistration.Create(PosixSignal.SIGTERM, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("LuckyPool mining: SIGTERM received — graceful shutdown");
        TryCancel(cts);
    });

    using var sigHup = PosixSignalRegistration.Create(PosixSignal.SIGHUP, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("LuckyPool mining: SIGHUP received — graceful shutdown");
        TryCancel(cts);
    });

    using var sigQuit = PosixSignalRegistration.Create(PosixSignal.SIGQUIT, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("LuckyPool mining: SIGQUIT received — graceful shutdown");
        TryCancel(cts);
    });

    AppDomain.CurrentDomain.ProcessExit += (_, _) => TryCancel(cts);

    using var shutdownDeadline = ShutdownDeadline.Arm(
        cts.Token,
        TimeSpan.FromSeconds(30),
        () => Environment.Exit(ShutdownDeadline.HardExitCode),
        log);

    if (opts.Observability.MetricsPort is int metricsPort)
        Metrics.TryStart(metricsPort, loggerFactory.CreateLogger("metrics"), cts.Token);

    await using var orchestrator = new WorkerOrchestrator(opts, loggerFactory);

    int attempt = 0;

    while (!cts.IsCancellationRequested)
    {
        var sessionStarted = Stopwatch.GetTimestamp();

        try
        {
            await using var client =
                new LuckyPoolClient(host, port, wallet, worker);

            await orchestrator
                .RunLuckyPoolMineAsync(client, cts.Token)
                .ConfigureAwait(false);

            // An uncancelled production session should not return normally,
            // but if it does, treat it as a transient disconnect.
            var ranFor = Stopwatch.GetElapsedTime(sessionStarted);
            attempt = ranFor >= TimeSpan.FromSeconds(30) ? 1 : attempt + 1;
        }
        catch (OperationCanceledException) when (cts.IsCancellationRequested)
        {
            break;
        }
        catch (Exception ex)
        {
            var ranFor = Stopwatch.GetElapsedTime(sessionStarted);
            attempt = ranFor >= TimeSpan.FromSeconds(30) ? 1 : attempt + 1;

            // Config/native compatibility problems do not improve by opening
            // another TCP connection. Fail fast rather than reconnect forever.
            if (ex is InvalidOperationException &&
                (ex.Message.Contains("No CUDA devices", StringComparison.OrdinalIgnoreCase) ||
                 ex.Message.Contains("does not support GPU", StringComparison.OrdinalIgnoreCase) ||
                 ex.Message.Contains("requires AKOYA_MINE_NOISE_RANK=128", StringComparison.OrdinalIgnoreCase)))
            {
                log.LogError(ex, "LuckyPool mining: fatal local configuration/native error");
                return 78;
            }

            log.LogWarning(
                ex,
                "LuckyPool session ended — reconnecting (attempt {Attempt})",
                attempt);
        }

        if (cts.IsCancellationRequested)
            break;

        var baseSeconds = attempt switch
        {
            <= 1 => 1.0,
            2    => 2.0,
            3    => 5.0,
            4    => 10.0,
            5    => 20.0,
            _    => 30.0,
        };

        // ±20% jitter prevents multiple rigs behind the same NAT from
        // reconnecting in lock-step after a pool restart.
        var jitter = 0.8 + Random.Shared.NextDouble() * 0.4;
        var delay = TimeSpan.FromSeconds(baseSeconds * jitter);

        log.LogInformation(
            "LuckyPool reconnect in {Delay:F1}s (attempt {Attempt})",
            delay.TotalSeconds,
            attempt);

        try
        {
            await Task.Delay(delay, cts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            break;
        }
    }

    log.LogInformation("LuckyPool mining: shutdown complete");
    return 0;
}

static int PerfIter(string[] args)
{
    using var loggerFactory = BuildLoggerFactory();
    var log = loggerFactory.CreateLogger("perf-iter");

    // perf-iter only needs the common mining/GPU configuration, but
    // EnvVarBindings.Load() also validates the normal pool identity fields.
    // Supply harmless local placeholders only when the operator did not set
    // those variables. No pool/network connection is opened by this command.
    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_POOL_WALLET")))
        Environment.SetEnvironmentVariable("AKOYA_POOL_WALLET", "perf-iter-local");
    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_POOL_WORKER")))
        Environment.SetEnvironmentVariable("AKOYA_POOL_WORKER", "perf-iter");

    // Do not silently use EnvVarBindings' generic fallback shape here. Normal
    // production startup lets WorkerOrchestrator replace that fallback with a
    // GPU-specific profile, but perf-iter calls GpuWorker directly. Requiring
    // M/N/K explicitly keeps the measurement reproducible and guarantees we
    // profile the exact shape the operator intends.
    if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_MINE_M")) ||
        string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_MINE_N")) ||
        string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AKOYA_MINE_K")))
    {
        log.LogError(
            "perf-iter requires explicit AKOYA_MINE_M, AKOYA_MINE_N and AKOYA_MINE_K values; " +
            "for RTX 5070 Ti baseline use M=4096 N=131072 K=4096.");
        return 78;
    }

    MinerOptions opts;
    try
    {
        opts = EnvVarBindings.Load(log);
    }
    catch (Exception ex)
    {
        log.LogError(ex, "perf-iter: configuration error");
        return 78;
    }

    double durationSec = 15.0;
    int runs = 1;

    if (args.Length > 1 &&
        (!double.TryParse(args[1], System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out durationSec) ||
         !double.IsFinite(durationSec) || durationSec <= 0.0 || durationSec > 300.0))
    {
        log.LogError("perf-iter: duration must be > 0 and <= 300 seconds; got '{Value}'", args[1]);
        return 64;
    }

    if (args.Length > 2 &&
        (!int.TryParse(args[2], System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture, out runs) ||
         runs < 1 || runs > 20))
    {
        log.LogError("perf-iter: runs must be in [1,20]; got '{Value}'", args[2]);
        return 64;
    }

    if (args.Length > 3)
    {
        log.LogError("usage: akoya-miner perf-iter [duration-sec] [runs]");
        return 64;
    }

    int deviceOrdinal;
    try
    {
        CudaDriver.Check(CudaDriver.Init(0), "cuInit");
        CudaDriver.Check(CudaDriver.DeviceGetCount(out var deviceCount), "cuDeviceGetCount");
        deviceOrdinal = ResolveSinglePerfGpu(opts.Gpus.IndicesRaw, deviceCount);
    }
    catch (Exception ex)
    {
        log.LogError(ex, "perf-iter: GPU selection failed");
        return 78;
    }

    // Force the raw-loop profiler to mpp=1 so one timed GPU batch equals one
    // mining iteration. All other mining options
    // (M/N/K/R, graph, ping/pong) remain exactly as configured by env vars.
    var mine = opts.Mine with { MatmulsPerPoll = 1 };

    log.LogWarning(
        "PERF-ITER — GPU={Gpu} shape M={M} N={N} K={K} R={R} mpp=1 graph={Graph} pong={Pong} duration={Sec:F1}s runs={Runs}; no pool connection, no shares",
        deviceOrdinal,
        mine.M,
        mine.N,
        mine.K,
        mine.NoiseRank,
        mine.CudaGraphIter,
        !mine.DisablePong,
        durationSec,
        runs);

    using var cts = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        try { cts.Cancel(); } catch (ObjectDisposedException) { }
    };

    try
    {
        for (int run = 1; run <= runs && !cts.IsCancellationRequested; run++)
        {
            var r = GpuWorker.RunIterLoopBenchmark(
                deviceOrdinal,
                mine,
                TimeSpan.FromSeconds(durationSec),
                loggerFactory.CreateLogger($"perf-iter-{deviceOrdinal}"),
                run,
                cts.Token);

            double tmadsPerSec =
                ((double)mine.M * mine.N * mine.K * r.ItersPerSec) / 1e12;

            log.LogWarning(
                "PERF-ITER RESULT run={Run}/{Runs}: iters={Iters} samples={Samples} sec={Sec:F3} iters/s={Ips:F3} iter_ms={IterMs:F3} tmads/s={Tmads:F3}",
                run, runs, r.IterCount, r.Samples, r.DurationSec,
                r.ItersPerSec, r.IterMs, tmadsPerSec);

            log.LogWarning(
                "PERF-ITER GPU: batch_p50={GpuP50:F3}ms batch_p95={GpuP95:F3}ms sync_p50={SyncP50:F3}ms sync_p95={SyncP95:F3}ms",
                r.GpuBatchP50Ms, r.GpuBatchP95Ms,
                r.SyncWaitP50Ms, r.SyncWaitP95Ms);

            log.LogWarning(
                "PERF-ITER HOST: loop_p50={LoopP50:F3}ms loop_p95={LoopP95:F3}ms queue_p50={QueueP50:F3}ms queue_p95={QueueP95:F3}ms host_nonwait_p50={HostP50:F3}ms scan_p50={ScanP50:F3}ms",
                r.LoopBatchWallP50Ms, r.LoopBatchWallP95Ms,
                r.QueueP50Ms, r.QueueP95Ms,
                r.HostNonWaitP50Ms, r.ScanP50Ms);

            log.LogWarning(
                "PERF-ITER ENQUEUE: header_clear_p50={HeaderClear:F3}ms header_ptr_p50={HeaderPtr:F3}ms device_clear_p50={DeviceClear:F3}ms iter_enqueue_p50={IterEnqueue:F3}ms graph_launch_p50={GraphLaunch:F3}ms graph_launches={GraphCount}",
                r.HeaderClearP50Ms, r.HeaderPtrP50Ms,
                r.DeviceClearEnqueueP50Ms, r.IterEnqueueP50Ms,
                r.GraphLaunchP50Ms, r.GraphLaunchCount);
        }

        return cts.IsCancellationRequested ? 130 : 0;
    }
    catch (OperationCanceledException) when (cts.IsCancellationRequested)
    {
        return 130;
    }
    catch (Exception ex)
    {
        log.LogError(ex, "perf-iter failed");
        return 1;
    }
}

static int ResolveSinglePerfGpu(string raw, int deviceCount)
{
    if (deviceCount <= 0)
        throw new InvalidOperationException("No CUDA devices found");

    raw = string.IsNullOrWhiteSpace(raw) ? "all" : raw.Trim();
    if (raw.Equals("all", StringComparison.OrdinalIgnoreCase))
    {
        if (deviceCount == 1) return 0;
        throw new InvalidOperationException(
            $"perf-iter profiles exactly one GPU; {deviceCount} CUDA devices are present. " +
            "Set AKOYA_GPU_INDICES=<single 0-based index>, e.g. AKOYA_GPU_INDICES=2.");
    }

    var tokens = raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    if (tokens.Length != 1 || !int.TryParse(tokens[0], out var ordinal))
        throw new FormatException(
            $"perf-iter requires exactly one AKOYA_GPU_INDICES value; got '{raw}'");

    if (ordinal < 0 || ordinal >= deviceCount)
        throw new ArgumentOutOfRangeException(
            nameof(raw),
            $"GPU index {ordinal} out of range [0, {deviceCount})");

    return ordinal;
}

static async Task<int> MineBlocksAsync(string[] _)
{
    using var loggerFactory = BuildLoggerFactory();
    var log = loggerFactory.CreateLogger("startup");

    MinerOptions opts;
    try { opts = EnvVarBindings.Load(log); }
    catch (Exception ex)
    {
        log.LogError(ex, "startup: configuration error");
        return 78; // EX_CONFIG
    }

    log.LogInformation("akoya-miner v{Ver} (git {Sha}) — pool={Host}:{Port} tls={Tls} tls_insecure={Insecure} wallet={Wallet} worker={Worker}",
        VersionInfo.MinerVersion, VersionInfo.GitSha,
        opts.Pool.Host, opts.Pool.Port, opts.Pool.UseTls, opts.Pool.TlsInsecure,
        opts.Pool.WalletAddress, opts.Pool.WorkerName);

    using var cts = new CancellationTokenSource();
    // Cancel-on-disposed-CTS guard: signal handlers and AppDomain.ProcessExit
    // can fire AFTER the `using var cts` scope has already disposed (e.g. when
    // a SIGINT arrives during the last ms of teardown, or when ProcessExit
    // runs as Main is unwinding). Without this, we'd crash with
    // ObjectDisposedException at the very moment we were about to exit
    // cleanly. Static local so all handlers below close over the same CTS.
    static void TryCancel(CancellationTokenSource c)
    {
        try { c.Cancel(); }
        catch (ObjectDisposedException) { /* race with normal shutdown — fine */ }
    }
    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        log.LogInformation("startup: Ctrl-C received — initiating graceful shutdown");
        TryCancel(cts);
    };
    // POSIX signal handling (HiveOS, systemd, k8s all send SIGTERM, not SIGINT).
    // PosixSignalRegistration intercepts BEFORE the runtime tears the process
    // down, so we get a real chance to drain. AppDomain.ProcessExit is kept
    // as a last-resort catch — it only fires AFTER unmanaged exit begins,
    // by which time `cts` has already been disposed by its `using` scope,
    // so the cancel call there will routinely race with disposal. Every
    // cancel site below uses TryCancel to tolerate that race instead of
    // bringing the process down with an ObjectDisposedException at the
    // very moment we were about to exit cleanly.
    using var sigTerm = PosixSignalRegistration.Create(PosixSignal.SIGTERM, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("startup: SIGTERM received — initiating graceful shutdown");
        TryCancel(cts);
    });
    using var sigHup = PosixSignalRegistration.Create(PosixSignal.SIGHUP, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("startup: SIGHUP received — initiating graceful shutdown");
        TryCancel(cts);
    });
    using var sigQuit = PosixSignalRegistration.Create(PosixSignal.SIGQUIT, ctx =>
    {
        ctx.Cancel = true;
        log.LogInformation("startup: SIGQUIT received — initiating graceful shutdown");
        TryCancel(cts);
    });
    AppDomain.CurrentDomain.ProcessExit += (_, _) => TryCancel(cts);

    // Shutdown deadline: after cancellation is requested, the rest of the
    // program MUST exit within 30s. If a CUDA handle is wedged or a native
    // teardown is stuck, we'd rather Environment.Exit ourselves than wait
    // for systemd/k8s/HiveOS to SIGKILL us mid-share-submit. Disposed at
    // the end of MineBlocksAsync, so a clean exit cancels the timer.
    //
    // 30s = worker DisposeGrace (10s) + pool channel shutdown (~2s) +
    // an in-flight share-submit allowance + slack. Tuned to land BELOW
    // every supervisor's default kill timer (k8s 30s default is a tight
    // squeeze — operators on k8s should raise terminationGracePeriodSeconds
    // to 60s if they care about clean shutdowns).
    using var shutdownDeadline = ShutdownDeadline.Arm(
        cts.Token,
        TimeSpan.FromSeconds(30),
        () => Environment.Exit(ShutdownDeadline.HardExitCode),
        log);

    if (opts.Observability.MetricsPort is int port)
    {
        Metrics.TryStart(port, loggerFactory.CreateLogger("metrics"), cts.Token);
    }

    // Reconnect loop: any unhandled stream exit (graceful, RpcException,
    // stream-watchdog cancellation, worker-watchdog cancellation) triggers a
    // jittered exponential backoff + Resume attempt. Fatal config errors
    // break out. Clean exits (server hangup, ReconnectHint) reconnect
    // immediately with attempt counter reset.
    int attempt = 0;
    // Construct the orchestrator ONCE per process. Inside, per-attempt
    // resources (PoolConnection, MiningSession, GpuWorkers) live in
    // RunAsync's using/await-using scopes and are recreated each loop.
    // What we deliberately keep across reconnects is orchestrator state
    // such as the cached benchmark result — the GPU rig's hashrate and
    // iter_ms don't change between a stream-end and the Resume that
    // follows, so re-benchmarking is wasted GPU time.
    var orchestrator = new WorkerOrchestrator(opts, loggerFactory);
    while (!cts.IsCancellationRequested)
    {
        TimeSpan? hintWait = null;
        try
        {
            await orchestrator.RunAsync(cts.Token).ConfigureAwait(false);
            log.LogInformation("orchestrator: stream ended cleanly — reconnecting");
            attempt = 0;
        }
        catch (OperationCanceledException) when (cts.IsCancellationRequested) { break; }
        catch (InvalidOperationException ex) when (ex.Message.Contains("Register rejected"))
        {
            log.LogError(ex, "fatal: server rejected registration — not retrying");
            return 78;
        }
        catch (PoolUnreachableException ex)
        {
            // Translated TaskCanceledException / RpcException(Unavailable|
            // DeadlineExceeded) from Register/Resume — channel never reached
            // ready state. Almost always wrong host/port or firewall. Skip
            // the stack trace (it's all Grpc internals) and surface just the
            // operator-actionable one-liner, then back off and retry like
            // any other transient failure.
            attempt++;
            var backoff = ReconnectBackoff.ComputeDelay(
                attempt, (Random.Shared.NextDouble() * 2) - 1);
            log.LogWarning(
                "orchestrator: {Msg} — retry in {Delay:F1}s (attempt {Attempt})",
                ex.Message, backoff.TotalSeconds, attempt);
            try { await Task.Delay(backoff, cts.Token).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }
        catch (StreamIdleException ex)
        {
            // Distinct log line: "gateway is alive but silent" is a very
            // different operational signal from a generic RPC failure.
            // We deliberately don't bypass the backoff path — silent stream
            // = treat-as-failure-attempt, same exp backoff applies.
            attempt++;
            var backoff = ReconnectBackoff.ComputeDelay(
                attempt, (Random.Shared.NextDouble() * 2) - 1);
            log.LogWarning(
                "orchestrator: stream went silent ({Msg}) — retry in {Delay:F1}s (attempt {Attempt})",
                ex.Message, backoff.TotalSeconds, attempt);
            try { await Task.Delay(backoff, cts.Token).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }
        catch (WorkerTripException ex)
        {
            attempt++;
            var backoff = ReconnectBackoff.ComputeDelay(
                attempt, (Random.Shared.NextDouble() * 2) - 1);
            log.LogWarning(ex,
                "orchestrator: local worker trip ({Reason}) — retry in {Delay:F1}s (attempt {Attempt})",
                ex.Reason, backoff.TotalSeconds, attempt);
            try { await Task.Delay(backoff, cts.Token).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }
        catch (Exception ex)
        {
            attempt++;
            // Exponential cap + ±25% jitter
            var backoff = ReconnectBackoff.ComputeDelay(
                attempt, (Random.Shared.NextDouble() * 2) - 1);
            log.LogWarning(ex, "orchestrator: error — retry in {Delay:F1}s (attempt {Attempt})",
                backoff.TotalSeconds, attempt);
            try { await Task.Delay(backoff, cts.Token).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }
        finally
        {
            // Capture any server-supplied ReconnectHint before disposing.
            if (orchestrator.LastReconnectHint is { WaitSeconds: > 0 } h)
            {
                if (ReconnectBackoff.HintWasClamped(h.WaitSeconds))
                {
                    log.LogWarning(
                        "orchestrator: ReconnectHint wait={W}s clamped to {C}s",
                        h.WaitSeconds, ReconnectBackoff.MaxReconnectHintSeconds);
                }
                hintWait = ReconnectBackoff.ApplyHint(
                    h.WaitSeconds, (Random.Shared.NextDouble() * 2) - 1);
            }
            await orchestrator.DisposeAsync().ConfigureAwait(false);
        }

        if (hintWait is TimeSpan w && !cts.IsCancellationRequested)
        {
            log.LogInformation("orchestrator: honouring ReconnectHint wait={W:F1}s", w.TotalSeconds);
            try { await Task.Delay(w, cts.Token).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }
    }

    log.LogInformation("akoya-miner: shutdown complete");
    return 0;
}

static int PrintVersion()
{
    Console.WriteLine($"akoya-miner v{VersionInfo.MinerVersion} (git {VersionInfo.GitSha})");
    Console.WriteLine("V2 gRPC miner + LuckyPool Stratum mining support");
    return 0;
}

static int Usage(string c)
{
    Console.Error.WriteLine($"unknown subcommand: {c}");
    Console.Error.WriteLine("usage: akoya-miner [mine-blocks|luckypool-test|luckypool-mine-test|luckypool-submit-test|luckypool-mine|perf-iter|selftest|version]");
    Console.Error.WriteLine("  mine-blocks          Connect to Akoya V2 pool, register/resume, mine. (default)");
    Console.Error.WriteLine("  luckypool-test       Connect/auth to LuckyPool, print first mining.notify job, exit.");
    Console.Error.WriteLine("  luckypool-mine-test    Experimental GPU dry-run on LuckyPool jobs; candidates are NEVER submitted.");
    Console.Error.WriteLine("  luckypool-submit-test  LIVE test: submit exactly one locally validated current-job PlainProof, print pool result, exit.");
    Console.Error.WriteLine("  luckypool-mine         Continuous LuckyPool mining: validate + submit shares, reconnect automatically.");
    Console.Error.WriteLine("  perf-iter              Profile one GPU steady-state iteration loop; usage: perf-iter [duration-sec] [runs].");
    Console.Error.WriteLine("  selftest             Validate config + pool + native libs + session store; emit JSON; exit 0/1.");
    Console.Error.WriteLine("  version              Print git sha + miner version.");
    return 64;
}

// --selftest: ship-readiness check that an operator can run once after install
// to validate every wire is connected, then bail. Returns 0 if all probes
// pass; 1 if any failed. Always emits a JSON report on stdout so wrappers
// (HiveOS rig checks, k8s initContainers, Docker HEALTHCHECK) can parse.
//
// Probe list:
//   config         — env vars load into MinerOptions without throwing
//   crashdump_env  — DOTNET_DbgEnableMiniDump is set (warn-only, doesn't fail)
//   pearl_gemm_lib — libpearl_gemm_capi.so resolves & loads
//   pearl_mining_lib — libpearl_mining_capi.so resolves & loads
//   session_store  — configured path is writable + readable (round-trip)
//   pool_tcp       — TCP connect to pool host:port within 5s
static async Task<int> SelfTestAsync(string[] _)
{
    var probes = new List<SelfTestProbe>();

    // Use a null logger so the JSON on stdout isn't polluted with prose.
    var log = NullLogger.Instance;

    MinerOptions? opts = null;
    probes.Add(RunProbe("config", () =>
    {
        opts = EnvVarBindings.Load(log);
        return $"host={opts.Pool.Host} port={opts.Pool.Port} tls={opts.Pool.UseTls} wallet_len={opts.Pool.WalletAddress.Length}";
    }));

    probes.Add(RunProbe("crashdump_env", () =>
    {
        var e = Environment.GetEnvironmentVariable("DOTNET_DbgEnableMiniDump");
        if (e != "1")
            throw new InvalidOperationException(
                "DOTNET_DbgEnableMiniDump != '1' — set it in the launcher / Dockerfile / systemd unit. " +
                "Without it, .NET will not write a core dump on fatal exceptions and field diagnosis " +
                "is limited to last-fatal.log (plain text, no native frames).");
        return $"set=1 type={Environment.GetEnvironmentVariable("DOTNET_DbgMiniDumpType") ?? "(unset)"} " +
               $"name={Environment.GetEnvironmentVariable("DOTNET_DbgMiniDumpName") ?? "(unset)"}";
    }, warnOnly: true));

    probes.Add(RunProbe("pearl_gemm_lib", () =>
    {
        NativeLibrary.Free(NativeLibs.Load("AKOYA_PEARL_GEMM_LIB", NativeLibs.GemmFile));
        return Environment.GetEnvironmentVariable("AKOYA_PEARL_GEMM_LIB") ?? $"{NativeLibs.GemmFile} (resolved)";
    }));

    probes.Add(RunProbe("pearl_mining_lib", () =>
    {
        NativeLibrary.Free(NativeLibs.Load("AKOYA_PEARL_MINING_LIB", NativeLibs.MiningFile));
        return Environment.GetEnvironmentVariable("AKOYA_PEARL_MINING_LIB") ?? $"{NativeLibs.MiningFile} (resolved)";
    }));

    probes.Add(RunProbe("session_store", () =>
    {
        if (opts is null) throw new InvalidOperationException("config probe failed; session_store not attempted");
        var path = opts.Session.FilePath;
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var probePath = path + ".selftest";
        var sentinel = $"akoya-miner selftest {DateTime.UtcNow:o}";
        File.WriteAllText(probePath, sentinel);
        var read = File.ReadAllText(probePath);
        File.Delete(probePath);
        if (read != sentinel) throw new IOException($"session-store roundtrip mismatch at {probePath}");
        return $"path={path} writable=true";
    }));

    await Task.Run(async () =>
    {
        probes.Add(await RunProbeAsync("pool_tcp", async () =>
        {
            if (opts is null) throw new InvalidOperationException("config probe failed; pool_tcp not attempted");
            using var tcp = new TcpClient();
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            await tcp.ConnectAsync(opts.Pool.Host, opts.Pool.Port, cts.Token).ConfigureAwait(false);
            return $"connected {opts.Pool.Host}:{opts.Pool.Port}";
        }).ConfigureAwait(false));
    }).ConfigureAwait(false);

    // Emit JSON manually — keeps us AOT-clean (no reflection-based serializer).
    var sb = new StringBuilder();
    sb.Append("{\"version\":\"").Append(VersionInfo.MinerVersion).Append("\",");
    sb.Append("\"git_sha\":\"").Append(VersionInfo.GitSha).Append("\",");
    sb.Append("\"timestamp\":\"").Append(DateTime.UtcNow.ToString("o")).Append("\",");
    sb.Append("\"probes\":[");
    for (int i = 0; i < probes.Count; i++)
    {
        if (i > 0) sb.Append(',');
        var p = probes[i];
        sb.Append("{\"name\":\"").Append(p.Name).Append("\",");
        sb.Append("\"status\":\"").Append(p.Status).Append("\",");
        sb.Append("\"detail\":\"").Append(JsonEscape(p.Detail)).Append('"');
        sb.Append('}');
    }
    sb.Append("],");
    bool anyFailed = probes.Any(p => p.Status == "fail");
    sb.Append("\"overall\":\"").Append(anyFailed ? "fail" : "pass").Append("\"}");
    Console.WriteLine(sb.ToString());
    return anyFailed ? 1 : 0;
}

static SelfTestProbe RunProbe(string name, Func<string> fn, bool warnOnly = false)
{
    try { return new SelfTestProbe(name, "pass", fn()); }
    catch (Exception ex) { return new SelfTestProbe(name, warnOnly ? "warn" : "fail", ex.Message); }
}

static async Task<SelfTestProbe> RunProbeAsync(string name, Func<Task<string>> fn, bool warnOnly = false)
{
    try { return new SelfTestProbe(name, "pass", await fn().ConfigureAwait(false)); }
    catch (Exception ex) { return new SelfTestProbe(name, warnOnly ? "warn" : "fail", ex.Message); }
}

static string JsonEscape(string s)
{
    var sb = new StringBuilder(s.Length + 8);
    foreach (var c in s)
    {
        switch (c)
        {
            case '\\': sb.Append("\\\\"); break;
            case '"':  sb.Append("\\\""); break;
            case '\b': sb.Append("\\b"); break;
            case '\f': sb.Append("\\f"); break;
            case '\n': sb.Append("\\n"); break;
            case '\r': sb.Append("\\r"); break;
            case '\t': sb.Append("\\t"); break;
            default:
                if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4"));
                else sb.Append(c);
                break;
        }
    }
    return sb.ToString();
}

static ILoggerFactory BuildLoggerFactory()
{
    var levelEnv = Environment.GetEnvironmentVariable("AKOYA_LOG_LEVEL") ?? "Information";
    if (!Enum.TryParse<LogLevel>(levelEnv, ignoreCase: true, out var level))
        level = LogLevel.Information;
    var json = (Environment.GetEnvironmentVariable("AKOYA_LOG_JSON") ?? "0") is "1" or "true";

    return LoggerFactory.Create(builder =>
    {
        var b = builder.SetMinimumLevel(level);
        if (json)
        {
            b.AddJsonConsole(opts =>
            {
                opts.IncludeScopes      = false;
                opts.UseUtcTimestamp    = true;
                opts.TimestampFormat    = "yyyy-MM-ddTHH:mm:ss.fffZ";
                opts.JsonWriterOptions  = new System.Text.Json.JsonWriterOptions { Indented = false };
            });
        }
        else
        {
            b.AddSimpleConsole(opts =>
            {
                opts.SingleLine      = true;
                opts.TimestampFormat = "HH:mm:ss.fff ";
                opts.UseUtcTimestamp = false;
                opts.IncludeScopes   = false;
                opts.ColorBehavior   = LoggerColorBehavior.Disabled;
            });
        }
    });
}

internal static class CrashDumpHelpers
{
    /// <summary>
    /// Resolves the dump directory in priority order:
    /// 1. AKOYA_DUMP_DIR
    /// 2. $AKOYA_HOME/dumps
    /// 3. $HOME/.akoya/dumps
    /// 4. /tmp/akoya-dumps
    /// </summary>
    public static string ResolveDumpDir()
    {
        var d = Environment.GetEnvironmentVariable("AKOYA_DUMP_DIR");
        if (!string.IsNullOrEmpty(d)) return d;
        var home = Environment.GetEnvironmentVariable("AKOYA_HOME");
        if (!string.IsNullOrEmpty(home)) return Path.Combine(home, "dumps");
        var userHome = Environment.GetEnvironmentVariable("HOME") ?? "/tmp";
        var akoyaHome = Path.Combine(userHome, ".akoya");
        return Path.Combine(akoyaHome, "dumps");
    }
}

internal readonly record struct SelfTestProbe(string Name, string Status, string Detail);

// Native library resolution for the miner's P/Invoke libraries.
//   1. $<envVar> — explicit absolute path override.
//   2. next to the executable (AppContext.BaseDirectory) — the layout build.sh
//      produces, so `./out/akoya-miner` finds `./out/lib*.so` with no env setup.
//   3. the OS loader (LD_LIBRARY_PATH / system paths) as a last resort.
internal static class NativeLibs
{
    // Platform-specific filenames for the two P/Invoke libraries the build
    // stages next to the binary: lib*.so on Linux, *.dll on Windows.
    public static string GemmFile =>
        OperatingSystem.IsWindows() ? "pearl_gemm_capi.dll" : "libpearl_gemm_capi.so";
    public static string MiningFile =>
        OperatingSystem.IsWindows() ? "pearl_mining_capi.dll" : "libpearl_mining_capi.so";

    public static nint Load(string envVar, string fileName)
    {
        var p = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrEmpty(p)) return NativeLibrary.Load(p);
        var local = Path.Combine(AppContext.BaseDirectory, fileName);
        if (File.Exists(local)) return NativeLibrary.Load(local);
        return NativeLibrary.Load(fileName);
    }
}
