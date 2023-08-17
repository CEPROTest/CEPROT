package IntervalTreeInstance;

import java.util.*;


public class IntervalTree<T extends Comparable<? super T>> extends AbstractSet<Interval<T>> {


	TreeNode<T> root;


	int size;

	
	@Override
	public boolean add(Interval<T> interval){
		if (interval.isEmpty()) {
			return false;
		}
		int sizeBeforeOperation = size;
		root = TreeNode.addInterval(this, root, interval);
		return size == sizeBeforeOperation;
	}


	public Set<Interval<T>> query(T point){
		return TreeNode.query(root, point, new HashSet<Interval<T>>());
	}


	public Set<Interval<T>> query(Interval<T> interval){
		Set<Interval<T>> result = new HashSet<>();

		if (root == null || interval.isEmpty()) {
			return result;
		}
		TreeNode<T> node = root;
		while (node != null){
			if (interval.contains(node.midpoint)){
				result.addAll(node.increasing);
				TreeNode.rangeQueryLeft(node.left, interval, result);
				TreeNode.rangeQueryRight(node.right, interval, result);
				break;
			}
			if (interval.isLeftOf(node.midpoint)) {
				for (Interval<T> next: node.increasing){
					if (!interval.intersects(next)) {
						break;
					}
					result.add(next);
				}
				node = node.left;
			}
			else {
				for (Interval<T> next: node.decreasing){
					if (!interval.intersects(next)) {
						break;
					}
					result.add(next);
				}
				node = node.right;
			}
		}
		return result;
	}


	public boolean remove(Interval<T> interval){
		if (interval.isEmpty() || root == null) {
			return false;
		}
		int sizeBeforeOperation = size;
		root = TreeNode.removeInterval(this, root, interval);
		return size == sizeBeforeOperation;
	}






	@Override
	public Iterator<Interval<T>> iterator() {
		if (root == null){
			return Collections.emptyIterator();
		}
		else {
			final TreeNode.TreeNodeIterator it = root.iterator();
			return new Iterator<Interval<T>>() {
				@Override
				public void remove() {
					if (it.currentNode.increasing.size() == 1){
						root = TreeNode.removeInterval(IntervalTree.this, root, it.currentInterval);

						TreeNode<T> node = root;
						it.stack = new Stack<>();


						while (node != it.subtreeRoot){
							if (it.currentNode.midpoint.compareTo(node.midpoint) < 0) {
								it.stack.push(node);
								node = node.left;
							}
							else {
								node = node.right;
							}
						}
					} else {
						it.remove();
					}
				}

				@Override
				public boolean hasNext() {
					return it.hasNext();
				}

				@Override
				public Interval<T> next() {
					return it.next();
				}
			};
		}
	}





	@Override
	public int size(){
		return size;
	}


	@Override
	public void clear() {
		size = 0;
		root = null;
	}


	@Override
	public boolean contains(Object o) {
		if (root == null || o == null) {
			return false;
		}
		if (!(o instanceof Interval)) {
			return false;
		}
		Interval<T> query;
		query = (Interval<T>)o;
		TreeNode<T> node = root;
		while (node != null){
			if (query.contains(node.midpoint)){
				return node.increasing.contains(query);
			}
			if (query.isLeftOf(node.midpoint)){
				node = node.left;
			} else {
				node = node.right;
			}
		}

		return false;
	}
}