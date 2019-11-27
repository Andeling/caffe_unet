#ifndef CAFFE_UTIL_CONNECTED_COMPONENT_LABELING_H_
#define CAFFE_UTIL_CONNECTED_COMPONENT_LABELING_H_

#include <list>
#include <cstring>

namespace caffe {

template<typename KeyT, typename ValueT>
class TreeNode {

public:

  TreeNode(KeyT const &key, ValueT const &value = ValueT())
          : _key(key), _value(value), p_parent(NULL), _children(),
            p_root(this) {}

  ~TreeNode() {
    while (_children.size() > 0) _children.front()->reparent(p_parent);
  }

  KeyT const &key() const {
    return _key;
  }

  void setKey(KeyT const &key) {
    _key = key;
  }

  ValueT const &value() const {
    return _value;
  }

  void setValue(ValueT const &value) {
    _value = value;
  }

  TreeNode<KeyT,ValueT> *parent() {
    return p_parent;
  }

  std::list<TreeNode<KeyT,ValueT>*> const &children() const {
    return _children;
  }

  TreeNode<KeyT,ValueT> *root() {
    return p_root;
  }

  TreeNode<KeyT,ValueT> const *root() const {
    return p_root;
  }

  void reparent(TreeNode<KeyT,ValueT> *parent) {
    if (p_parent == parent) return;
    if (p_parent != NULL) p_parent->_removeChild(this);
    p_parent = parent;
    if (parent != NULL) parent->_addChild(this);
    _updateRoot();
  }

  void addChild(TreeNode<KeyT,ValueT> *child) {
    child->reparent(this);
  }

  void removeChild(TreeNode<KeyT,ValueT> *child) {
    child->reparent(NULL);
  }

private:

  void _removeChild(TreeNode<KeyT,ValueT> *child) {
    _children.remove(child);
  }

  void _addChild(TreeNode<KeyT,ValueT> *child) {
    _children.push_back(child);
  }

  void _updateRoot() {
    if (p_parent == NULL) p_root = this;
    else p_root = p_parent->p_root;
    for (typename std::list<TreeNode<KeyT,ValueT>*>::iterator it =
             _children.begin(); it != _children.end(); ++it)
        (*it)->_updateRoot();
  }

  KeyT _key;
  ValueT _value;

  TreeNode<KeyT,ValueT> *p_parent;
  std::list<TreeNode<KeyT,ValueT>*> _children;

  TreeNode<KeyT,ValueT> *p_root;

};

  /**
   * Connected component labeling for n-D data. This method takes up to two
   * input arrays (labels and weights) and computes the connected component
   * labeling of the semantic segmentation contained in labels. labels can
   * contain multiple classes. Adjacent pixels with different class labels
   * will be assigned different connected component labels. Input array shape
   * is deduced from the instancelabels' blob shape, so it must be properly
   * Reshape'd before passing it to this method.
   *
   * @param instancelabels  The output blob storing the instance labels
   * @param labels          The input array containing the semantic
   *                        segmentation.
   * @param weights         An optional weights array. Areas with weight zero
   *                        are ignored, i.e. treated as background.
   * @param label_axis      Axis with indices less than this number are treated
   *                        as independent samples. Axes greater than this
   *                        number are treated as spatial axes.
   * @param backgroundLabel Areas with this label are treated as background
   * @param hasIgnoreLabel  If true, all pixels with given ignoreLabel are
   *                        ignored, i.e. treated as background. This is
   *                        equivalent to setting their weight to zero.
   * @param ignoreLabel     If hasIgnoreLabel is true, pixels with this label
   *                        are ignored, i.e. treated as background.
   *
   * @return The number of connected components per input sample
   */
template <typename Dtype>
std::vector<int> connectedComponentLabeling(
    Blob<int> *instancelabels, Dtype const *labels,
    Dtype const *weights = NULL, int label_axis = 1,
    Dtype backgroundLabel = 0, bool hasIgnoreLabel = false,
    Dtype ignoreLabel = 0) {
  int outer_num = instancelabels->count(0, label_axis);
  int inner_num = instancelabels->count(label_axis + 1);
  std::vector<int> labelShape(instancelabels->shape());
  int nDims = labelShape.size() - label_axis - 1;
  CHECK_GE(nDims, 1)
      << "The selected label axis " << label_axis << " leaves no spatial "
      << "dimensions for connected component labeling";
  std::vector<int> shape(nDims), strides(nDims);
  shape[nDims - 1] = labelShape.back();
  strides[nDims - 1] = 1;
  for (int d = nDims - 2; d >= 0; --d) {
    shape[d] = labelShape[d + label_axis + 1];
    strides[d] = strides[d + 1] * labelShape[d + label_axis + 2];
  }
  std::memset(instancelabels->mutable_cpu_data(), 0,
              instancelabels->count() * sizeof(int));
  std::vector<int> nInstances(outer_num, 0);
  for (int i = 0; i < outer_num; ++i) {
    std::vector<TreeNode<int,int>*> lbl(1, NULL);
    int nextLabel = 1;
    Dtype const *labelArr = labels + i * inner_num;
    Dtype const *weightArr = (weights != NULL) ? weights + i * inner_num : NULL;
    int *instanceLabelArr = instancelabels->mutable_cpu_data() + i * inner_num;

    // Iterate over all pixels
    for (int j = 0; j < inner_num; ++j) {
      if (labelArr[j] == backgroundLabel ||
          (hasIgnoreLabel && labelArr[j] == ignoreLabel) ||
          (weights != NULL && weightArr[j] == Dtype(0))) continue;
      int val = static_cast<int>(instanceLabelArr[j]);
      std::vector<int> pos(nDims, 0);
      int tmp = j;
      for (int d = nDims - 1; d >= 0; --d) {
        pos[d] = tmp % shape[d];
        tmp /= shape[d];
      }
      // Iterate over all left-side neighbors
      for (int d = 0; d < nDims; ++d) {
        if (pos[d] - 1 < 0) continue;
        int nbVal = instanceLabelArr[j - strides[d]];
        if (nbVal == 0 || val == nbVal ||
            labelArr[j] != labelArr[j - strides[d]]) continue;
        if (val == 0) {
          instanceLabelArr[j] = nbVal;
          val = nbVal;
          continue;
        }
        if (lbl[val]->root()->key() < lbl[nbVal]->root()->key())
            lbl[nbVal]->root()->reparent(lbl[val]->root());
        else if (lbl[val]->root()->key() > lbl[nbVal]->root()->key())
            lbl[val]->root()->reparent(lbl[nbVal]->root());
      }
      if (val == 0) {
        instanceLabelArr[j] = nextLabel;
        lbl.push_back(new TreeNode<int,int>(nextLabel++));
      }
    }

    // Generate dense label mapping
    std::vector<int> labelMap(lbl.size(), 0);
    int currentLabel = 1;
    for (int k = 1; k < lbl.size(); ++k)
    {
      if (labelMap[lbl[k]->root()->key()] == 0)
          labelMap[lbl[k]->root()->key()] = currentLabel++;
      labelMap[k] = labelMap[lbl[k]->root()->key()];
    }
    nInstances[i] = currentLabel - 1;

    // Re-map preliminary labels to final labels
    for (int j = 0; j < inner_num; ++j)
        instanceLabelArr[j] = labelMap[instanceLabelArr[j]];

    for (size_t k = 0; k < lbl.size(); ++k) delete lbl[k];
  }
  return nInstances;
}

} // namespace caffe

#endif // CAFFE_UTIL_CONNECTED_COMPONENT_LABELING_H_
