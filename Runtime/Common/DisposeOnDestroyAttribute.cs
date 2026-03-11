using System;

namespace Tuanjie.NeuralDeformer
{
    [AttributeUsage(AttributeTargets.Field)]
    internal class DisposeOnDestroyAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Field)]
    internal class ListDisposeOnDestroyAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Field)]
    internal class GPUResourceDisposeOnDestroyAttribute : Attribute { }

    [AttributeUsage(AttributeTargets.Field)]
    internal class GPUInferenceResourceDisposeOnDestroyAttribute : GPUResourceDisposeOnDestroyAttribute { }

    [AttributeUsage(AttributeTargets.Field)]
    internal class GPUPostprocessingResourceDisposeOnDestroyAttribute : GPUResourceDisposeOnDestroyAttribute { }
}