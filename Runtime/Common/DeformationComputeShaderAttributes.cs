namespace Tuanjie.NeuralDeformer
{
    static class Uniforms
    {
        public static int _deformationWeight;
        public static int _alphaMaskInfluenceWeight;
        public static int _modelNormalizationMaxValue;
        public static int _modelNormalizationMinValue;

        public static int _colorBuffer;
        public static int _deformationDeltaBuffer;
        public static int _vertexMappingBuffer;
        public static int _vertexPositionsBuffer;

        public static int _meshTriangleIndexBuffer;
        public static int _meshAdjacencyTriangleIndexBuffer;
        public static int _meshAdjacencyTriangleIndexOffsetBuffer;
        public static int _meshAdjacencyTriangleIndexStrideBuffer;
        public static int _triangleNormalBufferRW;
        public static int _triangleNormalBufferRO;

        public static int _targetMeshBufferRW;

        public static int _vertexCount;
        public static int _triangleCount;
        public static int _vertexBufferStride;
        public static int _vertexPositionAttributeOffset;
        public static int _vertexNormalAttributeOffset;
        public static int _colorBufferOffset;
        public static int _colorBufferStride;
        public static int _alphaMaskedDeformationWeightEnabled;
    }

    static class Kernels
    {
        public static int ApplyDeformation;
        public static int ComputeTriangleNormals;
        public static int ComputeVertexNormals;
    }
}
