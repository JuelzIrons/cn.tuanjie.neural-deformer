using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using Unity.Collections;
using UnityEngine;

namespace Tuanjie.NeuralDeformer
{
    /// <summary>
    /// NeuralDeformerDatasetMetaInfo is a ScriptableObject that stores metadata about processed deformer dataset.
    /// </summary>
    public class NeuralDeformerDatasetMetaInfo : ScriptableObject
    {
        [SerializeField, ReadOnlyProperty]
        private int m_JointCount;

        [SerializeField, ReadOnlyProperty]
        private List<string> m_JointNames;

        [SerializeField, ReadOnlyProperty]
        private int m_VertexCount;

        [SerializeField, ReadOnlyProperty]
        private int m_UniqueVertexCount;

        [SerializeField, ReadOnlyProperty]
        private int[] m_VertexTable;

        [SerializeField, ReadOnlyProperty]
        private float m_MinDelta;

        [SerializeField, ReadOnlyProperty]
        private float m_MaxDelta;

        public List<string> JointNames => m_JointNames;
        public int JointCount => m_JointCount;
        public int VertexCount => m_VertexCount;
        public int UniqueVertexCount => m_UniqueVertexCount;
        public float MinDelta => m_MinDelta;
        public float MaxDelta => m_MaxDelta;

        /// <summary>
        /// Initialization.
        /// </summary>
        /// <param name="joints">The list of joint <see cref="Transform"/>.</param>
        /// <param name="vertexTable">The index mapping from true vertices to reduced ones.</param>
        /// <param name="uniqueVertexCount">The number of unique vertices in the mesh.</param>
        /// <param name="minDelta">The minimum mesh vertex displacement.</param>
        /// <param name="maxDelta">The maximum mesh vertex displacement.</param>
        internal void Init(List<Transform> joints, int[] vertexTable, int uniqueVertexCount, float minDelta, float maxDelta)
        {
            m_JointCount = joints.Count;
            m_JointNames = joints.ConvertAll(j => j.name);
            m_VertexCount = vertexTable.Length;
            m_UniqueVertexCount = uniqueVertexCount;
            m_VertexTable = vertexTable;
            m_MinDelta = minDelta;
            m_MaxDelta = maxDelta;
        }

        internal NativeArray<int> GetVertexTable()
        {
            return new NativeArray<int>(m_VertexTable, Allocator.Persistent);
        }

        /// <summary>
        /// Try getting the list of joint transforms in current scene.
        /// </summary>
        /// <param name="joints">The list of joint transforms.</param>
        /// <returns>(<see cref="bool"/>) If it is done without errors.</returns>
        internal bool TryGetJointTransformList(ref List<Transform> joints)
        {
            bool succ = true;

            joints ??= new List<Transform>();
            joints.Clear();

            foreach (var jointName in m_JointNames)
            {
                var joint = GameObject.Find(jointName);
                if (joint == null)
                {
                    Debug.LogError($"Joint {jointName} not found in the scene.");
                    succ = false;
                    break;
                }

                joints.Add(joint.transform);
            }

            if (!succ) joints.Clear();
            return succ;
        }
    }
}
