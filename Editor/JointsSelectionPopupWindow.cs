using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.UIElements;
using UnityEditor.UIElements;
using System.Linq;

namespace Tuanjie.NeuralDeformer
{
    internal class JointsSelectionPopupWindow : EditorWindow
    {
        internal event Action<List<Transform>> onJointsSelected;

        private Transform m_Root;
        private HashSet<Transform> m_Joints;
        private List<TreeViewItemData<Transform>> m_JointsTreeData;
        private Dictionary<Transform, int> m_TransformTable;

        private Label m_JointsSummary;
        private TreeView m_JointsTree;
        internal VisualElement root => rootVisualElement;

        private const string k_JointToggleClass = "joint-transform-toggle";
        private const string k_TreeViewItemToggleOnClass = "tree-view__item--toggle-on";

        internal void Init(Transform root, List<Transform> joints)
        {
            m_Root = root;
            m_Joints = new HashSet<Transform>(joints);
            m_TransformTable = new();

            if (m_Root == null)
            {
                m_JointsTreeData = new List<TreeViewItemData<Transform>>();
            }
            else
            {
                int inOrderId = 0;
                m_TransformTable[root] = inOrderId;
                m_JointsTreeData = new()
            {
                new TreeViewItemData<Transform>(inOrderId++, m_Root, GetDataSourceFromTransformsRecursive(m_Root, ref inOrderId))
            };
            }

            m_JointsTree.SetRootItems(m_JointsTreeData);
            m_JointsTree.Rebuild();
            m_JointsTree.ClearSelection();
            UpdateJointsSummary();
        }

        private List<TreeViewItemData<Transform>> GetDataSourceFromTransformsRecursive(Transform root, ref int inOrderId)
        {
            List<TreeViewItemData<Transform>> source = root.childCount > 0 ? new(root.childCount) : null;
            for (int i = 0; i < root.childCount; i++)
            {
                var child = root.GetChild(i);
                m_TransformTable[child] = inOrderId;
                var itemData = new TreeViewItemData<Transform>(inOrderId++, child, GetDataSourceFromTransformsRecursive(child, ref inOrderId));
                source.Add(itemData);
            }
            return source;
        }

        private void SelectAllJointsRecursive(Transform transform, ref HashSet<Transform> joints)
        {
            if (transform == null || joints == null)
                return;

            joints.Add(transform);
            for (int i = 0; i < transform.childCount; i++)
            {
                SelectAllJointsRecursive(transform.GetChild(i), ref joints);
            }
        }

        private void DeselectAllJointsRecursive(Transform transform, ref HashSet<Transform> joints)
        {
            if (transform == null || joints == null)
                return;

            joints.Remove(transform);
            for (int i = 0; i < transform.childCount; i++)
            {
                DeselectAllJointsRecursive(transform.GetChild(i), ref joints);
            }
        }

        internal static JointsSelectionPopupWindow Open(Transform root, List<Transform> joints)
        {
            var window = EditorWindow.GetWindow<JointsSelectionPopupWindow>(utility: true);
            window.titleContent = new GUIContent("Select Joints");
            window.Init(root, joints);
            window.ShowUtility();
            return window;
        }

        private void CreateGUI()
        {
            root.styleSheets.Add(Resources.Load<StyleSheet>("ButtonRow"));
            root.styleSheets.Add(Resources.Load<StyleSheet>("JointsSelection"));

            var jsVisualTree = Resources.Load<VisualTreeAsset>("JointsSelection");
            jsVisualTree.CloneTree(root);

            m_JointsSummary = root.Q<Label>("JointsSummary");

            m_JointsTree = root.Q<TreeView>("JointsTree");
            m_JointsTree.makeItem = () =>
            {
                var itemRow = new VisualElement();
                itemRow.style.flexDirection = FlexDirection.Row;

                var toggle = new Toggle();
                toggle.AddToClassList(k_JointToggleClass);

                var transformField = new ObjectField();
                transformField.objectType = typeof(Transform);

                itemRow.Add(toggle);
                itemRow.Add(transformField);
                return itemRow;
            };
            m_JointsTree.bindItem = (e, i) =>
            {
                var transform = m_JointsTree.GetItemDataForIndex<Transform>(i);
                var toggle = e.Q<Toggle>(className: k_JointToggleClass);
                var id = m_JointsTree.GetIdForIndex(i);

                toggle.name = i.ToString();
                toggle.text = $"({id})";
                toggle.SetValueWithoutNotify(m_Joints.Contains(transform));

                VisualElement treeViewItem = e;
                for (; !treeViewItem.ClassListContains(TreeView.itemUssClassName); treeViewItem = treeViewItem.parent) ;
                RefreshTreeViewItemUI(treeViewItem, toggle.value);

                toggle.RegisterValueChangedCallback(evt =>
                {

                    evt.StopImmediatePropagation();

                    int i = int.Parse(toggle.name);
                    var transform = m_JointsTree.GetItemDataForIndex<Transform>(i);

                    // Update the selection in the tool when the toggle is changed
                    if (evt.newValue)
                    {
                        m_Joints.Add(transform);
                    }
                    else
                    {
                        m_Joints.Remove(transform);
                    }

                    if (!m_JointsTree.selectedIndices.Contains(i))
                    {
                        m_JointsTree.SetSelection(i);
                    }

                    RefreshTreeViewItemUI(treeViewItem, evt.newValue);
                    UpdateJointsSummary();
                });

                var transformField = e.Q<ObjectField>();
                transformField.value = transform;
            };
            // Callback invoked when the user double clicks or presses Enter on selected item(s)
            m_JointsTree.itemsChosen += (it) =>
            {
                bool value = true;
                foreach (Transform transform in it)
                {
                    if (!m_Joints.Contains(transform))
                    {
                        value = false;
                        break;
                    }
                }

                value = !value; // Toggle the selection state
                foreach (Transform transform in it)
                {
                    if (value)
                    {
                        m_Joints.Add(transform);
                    }
                    else
                    {
                        m_Joints.Remove(transform);
                    }
                }

                m_JointsTree.RefreshItems();
                UpdateJointsSummary();
            };

            var allBtn = root.Q<Button>("AllButton");
            allBtn.RegisterCallback<ClickEvent>(evt =>
            {
                SetAllJointsSelected(true);
            });

            var noneBtn = root.Q<Button>("NoneButton");
            noneBtn.RegisterCallback<ClickEvent>(evt =>
            {
                SetAllJointsSelected(false);
            });

            var collapseBtn = root.Q<Button>("CollapseButton");
            collapseBtn.RegisterCallback<ClickEvent>(evt =>
            {
                m_JointsTree.CollapseAll();
            });

            var expandBtn = root.Q<Button>("ExpandButton");
            expandBtn.RegisterCallback<ClickEvent>(evt =>
            {
                m_JointsTree.ExpandAll();
            });

            var cancelBtn = root.Q<Button>("CancelButton");
            cancelBtn.RegisterCallback<ClickEvent>(evt =>
            {
                Close();
            });

            var applyBtn = root.Q<Button>("ApplyButton");
            applyBtn.RegisterCallback<ClickEvent>(evt =>
            {
                if (m_JointsTree != null && m_Joints != null)
                {
                    onJointsSelected?.Invoke(m_Joints.OrderBy(t => m_TransformTable[t]).ToList());
                }
                Close();
            });
        }

        private void RefreshTreeViewItemUI(VisualElement itemElement, bool isSelected)
        {
            if (itemElement == null)
                return;

            if (isSelected)
            {
                itemElement.AddToClassList(k_TreeViewItemToggleOnClass);
            }
            else
            {
                itemElement.RemoveFromClassList(k_TreeViewItemToggleOnClass);
            }
        }

        private void UpdateJointsSummary()
        {
            int n = m_JointsTree?.GetTreeCount() ?? 0;
            bool hideSummary = m_Joints == null || m_JointsTree == null || n == 0;

            m_JointsSummary.style.display = hideSummary ? DisplayStyle.None : DisplayStyle.Flex;
            m_JointsSummary.text = $"({m_Joints.Count}/{n} selected)";
            m_JointsSummary.tooltip = string.Join(",\n", m_Joints.Select(t => t.name));
        }

        private void SetAllJointsSelected(bool selected)
        {
            if (selected)
            {
                SelectAllJointsRecursive(m_Root, ref m_Joints);
            }
            else
            {
                DeselectAllJointsRecursive(m_Root, ref m_Joints);
            }

            if (m_JointsTree == null)
                return;

            m_JointsTree.RefreshItems();
            UpdateJointsSummary();
        }

        private void OnDestroy()
        {
            m_JointsTree?.ClearSelection();
            m_JointsTree?.Clear();
            m_JointsTree = null;
            m_Joints = null;
            m_Root = null;
        }
    }
}