using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace Akoya.Miner.Stratum;

public sealed class LuckyPoolClient : IAsyncDisposable
{
    private readonly string _host;
    private readonly int _port;
    private readonly string _wallet;
    private readonly string _worker;

    private TcpClient? _tcpClient;
    private NetworkStream? _stream;
    private StreamReader? _reader;
    private StreamWriter? _writer;

    private readonly ConcurrentDictionary<long, TaskCompletionSource<LuckyPoolSubmitResult>> _pendingRequests = new();
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private string? _latestJobId;
    private long _nextRequestId;

    public string? LatestJobId => Volatile.Read(ref _latestJobId);

    public LuckyPoolClient(
        string host,
        int port,
        string wallet,
        string worker)
    {
        _host = host;
        _port = port;
        _wallet = wallet;
        _worker = worker;
    }

    // ============================================================
    // Connection
    // ============================================================

    public async Task ConnectAsync(
        CancellationToken cancellationToken = default)
    {
        Console.WriteLine(
            $"LuckyPool: connecting to {_host}:{_port} ...");

        Volatile.Write(ref _latestJobId, null);

        _tcpClient = new TcpClient
        {
            NoDelay = true
        };

        await _tcpClient.ConnectAsync(
            _host,
            _port,
            cancellationToken);

        _stream = _tcpClient.GetStream();

        _reader = new StreamReader(
            _stream,
            new UTF8Encoding(false),
            detectEncodingFromByteOrderMarks: false,
            bufferSize: 64 * 1024,
            leaveOpen: true);

        _writer = new StreamWriter(
            _stream,
            new UTF8Encoding(false),
            bufferSize: 64 * 1024,
            leaveOpen: true)
        {
            AutoFlush = true,
            NewLine = "\n"
        };

        Console.WriteLine(
            "LuckyPool: TCP connected.");

        await AuthorizeAsync(
            cancellationToken);
    }

    // ============================================================
    // Authorization
    // ============================================================

    private async Task AuthorizeAsync(
        CancellationToken cancellationToken)
    {
        var id = Interlocked.Increment(
            ref _nextRequestId);

        Console.WriteLine(
            $"LuckyPool: authorizing wallet={Shorten(_wallet)} worker={_worker}");

        await SendAuthorizeAsync(
            id,
            cancellationToken);
    }

    /*
     * IMPORTANT:
     *
     * Akoya is published using .NET Native AOT.
     *
     * JsonSerializer.Serialize(object) would normally use
     * reflection-based serialization, which is disabled by the
     * Native AOT build.
     *
     * Therefore outgoing Stratum messages are generated using
     * Utf8JsonWriter.
     */

    private async Task SendAuthorizeAsync(
        long id,
        CancellationToken cancellationToken)
    {
        EnsureConnected();

        using var buffer =
            new MemoryStream();

        using (var jsonWriter =
               new Utf8JsonWriter(buffer))
        {
            jsonWriter.WriteStartObject();

            jsonWriter.WriteNumber(
                "id",
                id);

            jsonWriter.WriteString(
                "method",
                "mining.authorize");

            jsonWriter.WritePropertyName(
                "params");

            jsonWriter.WriteStartObject();

            jsonWriter.WriteString(
                "wallet",
                _wallet);

            jsonWriter.WriteString(
                "worker",
                _worker);

            jsonWriter.WriteEndObject();

            jsonWriter.WriteEndObject();

            jsonWriter.Flush();
        }

        var json =
            Encoding.UTF8.GetString(
                buffer.ToArray());

        Console.WriteLine(
            $"LuckyPool TX: {json}");

        await WriteJsonLineAsync(json, cancellationToken);
    }

    // ============================================================
    // Share submission
    // ============================================================

    public async Task<LuckyPoolSubmitResult> SubmitPlainProofAsync(
        string jobId,
        string plainProofBase64,
        CancellationToken cancellationToken = default)
    {
        EnsureConnected();

        if (string.IsNullOrWhiteSpace(jobId))
            throw new ArgumentException("jobId must be non-empty.", nameof(jobId));
        if (string.IsNullOrWhiteSpace(plainProofBase64))
            throw new ArgumentException("plainProofBase64 must be non-empty.", nameof(plainProofBase64));

        var id = Interlocked.Increment(ref _nextRequestId);
        var pending = new TaskCompletionSource<LuckyPoolSubmitResult>(
            TaskCreationOptions.RunContinuationsAsynchronously);

        if (!_pendingRequests.TryAdd(id, pending))
            throw new InvalidOperationException($"Duplicate LuckyPool request id {id}.");

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(30));
        using var reg = timeoutCts.Token.Register(() => pending.TrySetCanceled(timeoutCts.Token));

        try
        {
            await SendSubmitAsync(id, jobId, plainProofBase64, cancellationToken);
            return await pending.Task.ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException(
                $"Timed out waiting for LuckyPool mining.submit response (id={id}, job={jobId}).");
        }
        finally
        {
            _pendingRequests.TryRemove(id, out _);
        }
    }

    private async Task SendSubmitAsync(
        long id,
        string jobId,
        string plainProofBase64,
        CancellationToken cancellationToken)
    {
        EnsureConnected();

        using var buffer = new MemoryStream();
        using (var jsonWriter = new Utf8JsonWriter(buffer))
        {
            jsonWriter.WriteStartObject();
            jsonWriter.WriteNumber("id", id);
            jsonWriter.WriteString("method", "mining.submit");
            jsonWriter.WritePropertyName("params");
            jsonWriter.WriteStartObject();
            jsonWriter.WriteString("wallet", _wallet);
            jsonWriter.WriteString("worker", _worker);
            jsonWriter.WriteString("job_id", jobId);
            jsonWriter.WriteString("plain_proof", plainProofBase64);
            jsonWriter.WriteEndObject();
            jsonWriter.WriteEndObject();
            jsonWriter.Flush();
        }

        var json = Encoding.UTF8.GetString(buffer.ToArray());

        // Never print the full proof (~hundreds of KiB) into HiveOS logs.
        Console.WriteLine(
            $"LuckyPool TX: mining.submit id={id} job={jobId} proofBase64Chars={plainProofBase64.Length}");

        await WriteJsonLineAsync(json, cancellationToken);
    }

    private async Task WriteJsonLineAsync(string json, CancellationToken cancellationToken)
    {
        EnsureConnected();
        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await _writer!.WriteLineAsync(json.AsMemory(), cancellationToken).ConfigureAwait(false);
            await _writer.FlushAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    // ============================================================
    // Job receive loop
    // ============================================================

    public async Task<LuckyPoolJob> WaitForJobAsync(
        CancellationToken cancellationToken = default)
    {
        EnsureConnected();

        Console.WriteLine(
            "LuckyPool: waiting for mining.notify ...");

        while (!cancellationToken.IsCancellationRequested)
        {
            var line =
                await _reader!.ReadLineAsync(
                    cancellationToken);

            if (line is null)
            {
                throw new IOException(
                    "LuckyPool closed the connection.");
            }

            if (string.IsNullOrWhiteSpace(line))
                continue;

            Console.WriteLine();
            Console.WriteLine(
                $"LuckyPool RX: {line}");

            JsonDocument document;

            try
            {
                document =
                    JsonDocument.Parse(line);
            }
            catch (JsonException ex)
            {
                Console.WriteLine(
                    $"LuckyPool: invalid JSON ignored: {ex.Message}");

                continue;
            }

            using (document)
            {
                var root =
                    document.RootElement;

                // ====================================================
                // Response to a request (e.g. mining.authorize)
                // ====================================================

                if (root.TryGetProperty(
                        "id",
                        out var idElement) &&
                    idElement.ValueKind != JsonValueKind.Null &&
                    root.TryGetProperty(
                        "result",
                        out var resultElement))
                {
                    var errorJson =
                        root.TryGetProperty("error", out var errorElement) &&
                        errorElement.ValueKind != JsonValueKind.Null
                            ? errorElement.GetRawText()
                            : null;

                    Console.WriteLine(
                        $"LuckyPool: response id={idElement} result={resultElement}" +
                        (errorJson is null ? "" : $" error={errorJson}"));

                    if (idElement.TryGetInt64(out var responseId) &&
                        _pendingRequests.TryGetValue(responseId, out var pending))
                    {
                        var accepted = resultElement.ValueKind == JsonValueKind.True;
                        pending.TrySetResult(new LuckyPoolSubmitResult(
                            RequestId: responseId,
                            Accepted: accepted,
                            ResultJson: resultElement.GetRawText(),
                            ErrorJson: errorJson));
                        continue;
                    }

                    // Non-pending response here is the authorize ACK.
                    if (resultElement.ValueKind == JsonValueKind.False)
                    {
                        throw new InvalidOperationException(
                            $"LuckyPool request id={idElement} was rejected: {errorJson ?? "(no error)"}");
                    }

                    continue;
                }

                // ====================================================
                // Notification
                // ====================================================

                if (!root.TryGetProperty(
                        "method",
                        out var methodElement))
                {
                    Console.WriteLine(
                        "LuckyPool: JSON message without method — ignored.");

                    continue;
                }

                var method =
                    methodElement.GetString();

                Console.WriteLine(
                    $"LuckyPool: method={method}");

                // ====================================================
                // Difficulty update
                // ====================================================

                if (method ==
                    "mining.set_difficulty")
                {
                    if (root.TryGetProperty(
                            "params",
                            out var difficulty))
                    {
                        Console.WriteLine(
                            $"LuckyPool: difficulty update: {difficulty}");
                    }

                    continue;
                }

                // ====================================================
                // Ignore unknown notifications
                // ====================================================

                if (method !=
                    "mining.notify")
                {
                    Console.WriteLine(
                        $"LuckyPool: ignoring unsupported method '{method}'.");

                    continue;
                }

                // ====================================================
                // mining.notify
                // ====================================================

                if (!root.TryGetProperty(
                        "params",
                        out var parameters))
                {
                    Console.WriteLine(
                        "LuckyPool: mining.notify without params.");

                    continue;
                }

                var headerHex =
                    GetRequiredString(
                        parameters,
                        "header");

                var targetHex =
                    GetRequiredString(
                        parameters,
                        "target");

                var jobId =
                    GetRequiredString(
                        parameters,
                        "job_id");

                ulong? height = null;

                if (parameters.TryGetProperty(
                        "height",
                        out var heightElement))
                {
                    if (heightElement.ValueKind ==
                            JsonValueKind.Number &&
                        heightElement.TryGetUInt64(
                            out var parsedHeight))
                    {
                        height =
                            parsedHeight;
                    }
                }

                // ====================================================
                // Header validation
                // ====================================================

                byte[] headerBytes;

                try
                {
                    headerBytes =
                        Convert.FromHexString(
                            headerHex);
                }
                catch (FormatException ex)
                {
                    throw new InvalidDataException(
                        "LuckyPool returned an invalid hex header.",
                        ex);
                }

                if (headerBytes.Length != 76)
                {
                    throw new InvalidDataException(
                        $"Expected 76-byte Pearl header, " +
                        $"received {headerBytes.Length} bytes.");
                }

                // ====================================================
                // Target validation
                // ====================================================

                var normalizedTarget =
                    targetHex.StartsWith(
                        "0x",
                        StringComparison.OrdinalIgnoreCase)
                        ? targetHex[2..]
                        : targetHex;

                normalizedTarget =
                    normalizedTarget.PadLeft(
                        64,
                        '0');

                if (normalizedTarget.Length > 64)
                {
                    throw new InvalidDataException(
                        $"LuckyPool target is larger than 256 bits: " +
                        $"{normalizedTarget.Length * 4} bits.");
                }

                byte[] targetBytes;

                try
                {
                    targetBytes =
                        Convert.FromHexString(
                            normalizedTarget);
                }
                catch (FormatException ex)
                {
                    throw new InvalidDataException(
                        "LuckyPool returned an invalid hex target.",
                        ex);
                }

                // ====================================================
                // Build internal job representation
                // ====================================================

                var job =
                    new LuckyPoolJob(
                        JobId: jobId,
                        Height: height,
                        HeaderHex: headerHex,
                        HeaderBytes: headerBytes,
                        TargetHex: normalizedTarget,
                        TargetBytes: targetBytes);

                Volatile.Write(ref _latestJobId, job.JobId);
                PrintJob(job);

                return job;
            }
        }

        throw new OperationCanceledException(
            cancellationToken);
    }

    // ============================================================
    // Continuous job loop (used by GPU dry-run / future Stratum mining)
    // ============================================================

    public async Task RunJobLoopAsync(
        Func<LuckyPoolJob, ValueTask> onJob,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(onJob);

        try
        {
            await ConnectAsync(cancellationToken).ConfigureAwait(false);

            while (!cancellationToken.IsCancellationRequested)
            {
                var job = await WaitForJobAsync(cancellationToken).ConfigureAwait(false);
                await onJob(job).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            FailPendingRequests(new OperationCanceledException(
                "LuckyPool receive loop cancelled.", cancellationToken));
            throw;
        }
        catch (Exception ex)
        {
            // Any socket/read/parser failure invalidates every request that is
            // waiting for a response on this connection. Wake those submitters
            // immediately instead of leaving their finalizer threads blocked
            // until the per-request 30s timeout.
            FailPendingRequests(ex);
            throw;
        }
    }

    // ============================================================
    // Test mode
    // ============================================================

    public async Task RunUntilFirstJobAsync(
        CancellationToken cancellationToken = default)
    {
        await ConnectAsync(
            cancellationToken);

        var job =
            await WaitForJobAsync(
                cancellationToken);

        Console.WriteLine();
        Console.WriteLine(
            "=============================================");
        Console.WriteLine(
            " LuckyPool real mining job successfully read ");
        Console.WriteLine(
            "=============================================");
        Console.WriteLine();

        PrintJob(job);
    }

    private void FailPendingRequests(Exception error)
    {
        foreach (var pair in _pendingRequests)
        {
            if (_pendingRequests.TryRemove(pair.Key, out var pending))
                pending.TrySetException(error);
        }
    }

    // ============================================================
    // Helpers
    // ============================================================

    private void EnsureConnected()
    {
        if (_tcpClient is null ||
            _stream is null ||
            _reader is null ||
            _writer is null)
        {
            throw new InvalidOperationException(
                "LuckyPoolClient is not connected.");
        }
    }

    private static string GetRequiredString(
        JsonElement element,
        string property)
    {
        if (!element.TryGetProperty(
                property,
                out var value))
        {
            throw new InvalidDataException(
                $"Missing required property '{property}'.");
        }

        if (value.ValueKind !=
            JsonValueKind.String)
        {
            throw new InvalidDataException(
                $"Property '{property}' is not a string.");
        }

        var result =
            value.GetString();

        if (string.IsNullOrWhiteSpace(
                result))
        {
            throw new InvalidDataException(
                $"Property '{property}' is empty.");
        }

        return result;
    }

    private static void PrintJob(
        LuckyPoolJob job)
    {
        Console.WriteLine();
        Console.WriteLine(
            "---------- LuckyPool JOB ----------");

        Console.WriteLine(
            $"job_id       : {job.JobId}");

        Console.WriteLine(
            $"height       : {job.Height?.ToString() ?? "unknown"}");

        Console.WriteLine(
            $"header       : {job.HeaderHex}");

        Console.WriteLine(
            $"header bytes : {job.HeaderBytes.Length}");

        Console.WriteLine(
            $"target       : {job.TargetHex}");

        Console.WriteLine(
            $"target bytes : {job.TargetBytes.Length}");

        Console.WriteLine(
            "-----------------------------------");

        Console.WriteLine();
    }

    private static string Shorten(
        string value)
    {
        if (value.Length <= 16)
            return value;

        return
            value[..8] +
            "..." +
            value[^6..];
    }

    // ============================================================
    // Cleanup
    // ============================================================

    public async ValueTask DisposeAsync()
    {
        FailPendingRequests(
            new IOException("LuckyPool client disposed before request completed."));

        Volatile.Write(ref _latestJobId, null);

        try
        {
            if (_writer is not null)
                await _writer.DisposeAsync().ConfigureAwait(false);
        }
        catch
        {
            // Disposal is best-effort; the connection is already unusable.
        }

        try { _reader?.Dispose(); } catch { }
        try { _stream?.Dispose(); } catch { }
        try { _tcpClient?.Dispose(); } catch { }
        _writeLock.Dispose();
    }
}

public sealed record LuckyPoolSubmitResult(
    long RequestId,
    bool Accepted,
    string ResultJson,
    string? ErrorJson);

// ================================================================
// LuckyPool job representation
// ================================================================

public sealed record LuckyPoolJob(
    string JobId,
    ulong? Height,
    string HeaderHex,
    byte[] HeaderBytes,
    string TargetHex,
    byte[] TargetBytes);
