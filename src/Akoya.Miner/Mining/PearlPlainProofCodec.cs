using System.Runtime.InteropServices;
using Akoya.Mining;
using Google.Protobuf;
using PearlPool.Proto.V2;

namespace Akoya.Miner.Mining;

internal readonly record struct PlainProofEncoding(string Base64, int ByteLength);

/// <summary>
/// Dense Pearl PlainProof encoder used by the LuckyPool Stratum path.
///
/// The native side performs two checks before returning bytes:
///  1. A and B^T Merkle proofs reconstruct HashA / HashB under the job key.
///  2. The Merkle leaf-index sets exactly match the supplied matrix row indices.
///
/// Only then is the proof encoded in the current Pearl dense PlainProof bincode
/// layout. Base64 is done in managed code after the validated binary blob returns.
/// </summary>
internal static class PearlPlainProofCodec
{
    public static unsafe PlainProofEncoding EncodeDenseAndValidate(
        ShareSubmission share,
        ReadOnlySpan<byte> jobKey,
        uint[] aRowIndices,
        uint[] bColIndices,
        int m,
        int n,
        int k,
        int noiseRank)
    {
        ArgumentNullException.ThrowIfNull(share);
        ArgumentNullException.ThrowIfNull(aRowIndices);
        ArgumentNullException.ThrowIfNull(bColIndices);

        if (jobKey.Length != 32)
            throw new ArgumentException("jobKey must be exactly 32 bytes.", nameof(jobKey));
        if (share.HashA.Length != 32 || share.HashB.Length != 32)
            throw new InvalidOperationException("Share HashA/HashB must both be 32 bytes.");
        if (m <= 0 || n <= 0 || k <= 0 || noiseRank <= 0)
            throw new ArgumentOutOfRangeException(nameof(m), "PlainProof dimensions/rank must be positive.");
        if (aRowIndices.Length == 0 || bColIndices.Length == 0)
            throw new InvalidOperationException("PlainProof row-index lists must be non-empty.");

        var aLeafData = FlattenFixedChunks(share.AProof.LeafData, 1024, "A leaf_data");
        var aLeafIndices = share.AProof.LeafIndices.ToArray();
        var aSiblings = FlattenFixedChunks(share.AProof.Siblings, 32, "A siblings");

        var bLeafData = FlattenFixedChunks(share.BProof.LeafData, 1024, "B leaf_data");
        var bLeafIndices = share.BProof.LeafIndices.ToArray();
        var bSiblings = FlattenFixedChunks(share.BProof.Siblings, 32, "B siblings");

        if (aLeafIndices.Length != share.AProof.LeafData.Count)
            throw new InvalidOperationException("A proof leaf_indices/leaf_data count mismatch.");
        if (bLeafIndices.Length != share.BProof.LeafData.Count)
            throw new InvalidOperationException("B proof leaf_indices/leaf_data count mismatch.");

        byte[] key = jobKey.ToArray();
        byte[] hashA = share.HashA.ToByteArray();
        byte[] hashB = share.HashB.ToByteArray();

        nint outPtr = 0;
        nuint outLen = 0;
        nint errPtr = 0;

        fixed (byte* pKey = key)
        fixed (byte* pALeafData = aLeafData)
        fixed (uint* pALeafIndices = aLeafIndices)
        fixed (uint* pARows = aRowIndices)
        fixed (byte* pARoot = hashA)
        fixed (byte* pASiblings = aSiblings)
        fixed (byte* pBLeafData = bLeafData)
        fixed (uint* pBLeafIndices = bLeafIndices)
        fixed (uint* pBRows = bColIndices)
        fixed (byte* pBRoot = hashB)
        fixed (byte* pBSiblings = bSiblings)
        {
            int rc = PlainProofEncodeDense(
                (nuint)m,
                (nuint)n,
                (nuint)k,
                (nuint)noiseRank,
                pKey,
                pALeafData,
                (nuint)share.AProof.LeafData.Count,
                pALeafIndices,
                (nuint)aLeafIndices.Length,
                pARows,
                (nuint)aRowIndices.Length,
                (nuint)share.AProof.TotalLeaves,
                pARoot,
                pASiblings,
                (nuint)share.AProof.Siblings.Count,
                pBLeafData,
                (nuint)share.BProof.LeafData.Count,
                pBLeafIndices,
                (nuint)bLeafIndices.Length,
                pBRows,
                (nuint)bColIndices.Length,
                (nuint)share.BProof.TotalLeaves,
                pBRoot,
                pBSiblings,
                (nuint)share.BProof.Siblings.Count,
                out outPtr,
                out outLen,
                out errPtr);

            if (rc != 0)
            {
                string message = ReadAndFreeError(errPtr);
                throw new InvalidOperationException(
                    $"Pearl PlainProof encode/validate failed (rc={rc}): {message}");
            }
        }

        if (outPtr == 0 || outLen == 0)
            throw new InvalidOperationException("Pearl PlainProof encoder returned an empty buffer.");
        if (outLen > int.MaxValue)
        {
            FreeBuffer(outPtr, outLen);
            throw new InvalidOperationException($"Pearl PlainProof buffer too large: {outLen} bytes.");
        }

        try
        {
            var bytes = new byte[(int)outLen];
            Marshal.Copy(outPtr, bytes, 0, bytes.Length);
            return new PlainProofEncoding(Convert.ToBase64String(bytes), bytes.Length);
        }
        finally
        {
            FreeBuffer(outPtr, outLen);
        }
    }

    private static byte[] FlattenFixedChunks(
        IEnumerable<ByteString> chunks,
        int chunkSize,
        string label)
    {
        var list = chunks as ICollection<ByteString> ?? chunks.ToArray();
        if (list.Count == 0)
            throw new InvalidOperationException($"{label} must be non-empty.");

        var result = new byte[checked(list.Count * chunkSize)];
        int offset = 0;
        foreach (var chunk in list)
        {
            if (chunk.Length != chunkSize)
                throw new InvalidOperationException(
                    $"{label} chunk has {chunk.Length} bytes; expected {chunkSize}.");
            chunk.Span.CopyTo(result.AsSpan(offset, chunkSize));
            offset += chunkSize;
        }
        return result;
    }

    private static string ReadAndFreeError(nint errPtr)
    {
        if (errPtr == 0)
            return "native call failed without an error message";

        try
        {
            return Marshal.PtrToStringUTF8(errPtr) ?? "(unreadable native error)";
        }
        finally
        {
            FreeString(errPtr);
        }
    }

    [DllImport(PearlMiningNative.Lib, EntryPoint = "pearl_capi_plain_proof_encode_dense")]
    private static extern unsafe int PlainProofEncodeDense(
        nuint m,
        nuint n,
        nuint k,
        nuint noiseRank,
        byte* keyPtr,
        byte* aLeafDataPtr,
        nuint aLeafCount,
        uint* aLeafIndicesPtr,
        nuint aLeafIndicesLen,
        uint* aRowIndicesPtr,
        nuint aRowIndicesLen,
        nuint aTotalLeaves,
        byte* aRootPtr,
        byte* aSiblingsPtr,
        nuint aSiblingCount,
        byte* bLeafDataPtr,
        nuint bLeafCount,
        uint* bLeafIndicesPtr,
        nuint bLeafIndicesLen,
        uint* bRowIndicesPtr,
        nuint bRowIndicesLen,
        nuint bTotalLeaves,
        byte* bRootPtr,
        byte* bSiblingsPtr,
        nuint bSiblingCount,
        out nint outBytes,
        out nuint outLen,
        out nint errMsgPtr);

    [DllImport(PearlMiningNative.Lib, EntryPoint = "pearl_capi_free_buffer")]
    private static extern void FreeBuffer(nint ptr, nuint len);

    [DllImport(PearlMiningNative.Lib, EntryPoint = "pearl_capi_free_string")]
    private static extern void FreeString(nint ptr);
}
