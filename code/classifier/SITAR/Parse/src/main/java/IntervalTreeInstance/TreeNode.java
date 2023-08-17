package IntervalTreeInstance;

import java.util.*;


public class TreeNode<T extends Comparable<? super T>> implements Iterable<Interval<T>> {

	protected NavigableSet<Interval<T>> increasing;


	protected NavigableSet<Interval<T>> decreasing;


	protected TreeNode<T> left;

	protected TreeNode<T> right;


	protected final T midpoint;

	/**
	 * The height of the node.
	 */
	protected int height;


	public TreeNode(Interval<T> interval){
		decreasing = new TreeSet<>(Interval.sweepRightToLeft);
		increasing = new TreeSet<>(Interval.sweepLeftToRight);

		decreasing.add(interval);
		increasing.add(interval);
		midpoint = interval.getMidpoint();
		height = 1;
	}

	
	public static <T extends Comparable<? super T>> TreeNode<T> addInterval(IntervalTree<T> tree, TreeNode<T> root, Interval<T> interval) {
		if (root == null) {
			tree.size++;
			return new TreeNode<>(interval);
		}
		if (interval.contains(root.midpoint)){
			if (root.decreasing.add(interval))
				tree.size++;
			root.increasing.add(interval);
			return root;
		} else if (interval.isLeftOf(root.midpoint)){
			root.left = addInterval(tree, root.left, interval);
			root.height = Math.max(height(root.left), height(root.right))+1;
		} else {
			root.right = addInterval(tree, root.right, interval);
			root.height = Math.max(height(root.left), height(root.right))+1;
		}

		return root.balanceOut();
	}


	public int height(){
		return height;
	}


	private static int height(TreeNode node){
		return node == null ? 0 : node.height();
	}


	private TreeNode<T> balanceOut(){
		int balance = height(left) - height(right);
		if (balance < -1){
			if (height(right.left) > height(right.right)){
				this.right = this.right.rightRotate();
				return leftRotate();
			} else{
				return leftRotate();
			}
		} else if (balance > 1){
			if (height(left.right) > height(left.left)){
				this.left = this.left.leftRotate();
				return rightRotate();
			} else
				return rightRotate();
		} else {
			return this;
		}
	}

	
	private TreeNode<T> leftRotate(){
		TreeNode<T> head = right;
		right = head.left;
		head.left = this;
		height = Math.max(height(right), height(left)) + 1;
		head.left = head.assimilateOverlappingIntervals(this);
		return head;
	}


	private TreeNode<T> rightRotate(){
		TreeNode<T> head = left;
		left = head.right;
		head.right = this;
		height = Math.max(height(right), height(left)) + 1;
		head.right = head.assimilateOverlappingIntervals(this);
		return head;
	}

	
	private TreeNode<T> assimilateOverlappingIntervals(TreeNode<T> from) {
		ArrayList<Interval<T>> tmp = new ArrayList<>();

		if (midpoint.compareTo(from.midpoint) < 0){
			for (Interval<T> next: from.increasing){
				if (next.isRightOf(midpoint))
					break;
				tmp.add(next);
			}
		} else {
			for (Interval<T> next: from.decreasing){
				if (next.isLeftOf(midpoint))
					break;
				tmp.add(next);
			}
		}

		from.increasing.removeAll(tmp);
		from.decreasing.removeAll(tmp);
		increasing.addAll(tmp);
		decreasing.addAll(tmp);
		if (from.increasing.size() == 0){
			return deleteNode(from);
		}
		return from;
	}

	
	public static <T extends Comparable<? super T>> Set<Interval<T>> query(TreeNode<T> root, T point, Set<Interval<T>> res) {
		if (root == null)
			return res;
		if (point.compareTo(root.midpoint) <= 0){
			for (Interval<T> next: root.increasing){
				if (next.isRightOf(point))
					break;
				res.add(next);
			}
			return TreeNode.query(root.left, point, res);
		} else{
			for (Interval<T> next: root.decreasing){
				if (next.isLeftOf(point))
					break;
				res.add(next);
			}
			return TreeNode.query(root.right, point, res);
		}
	}


	
	public static <T extends Comparable<? super T>> TreeNode<T> removeInterval(IntervalTree<T> tree, TreeNode<T> root, Interval<T> interval) {
		if (root == null)
			return null;
		if (interval.contains(root.midpoint)){
			if (root.decreasing.remove(interval))
				tree.size--;
			root.increasing.remove(interval);
			if (root.increasing.size() == 0){
				return deleteNode(root);
			}

		} else if (interval.isLeftOf(root.midpoint)){
			root.left = removeInterval(tree, root.left, interval);
		} else {
			root.right = removeInterval(tree, root.right, interval);
		}
		return root.balanceOut();
	}

	
	private static <T extends Comparable<? super T>> TreeNode<T> deleteNode(TreeNode<T> root) {
		if (root.left == null && root.right == null)
			return null;

		if (root.left == null){
			
			return root.right;
		} else {
			TreeNode<T> node = root.left;
			Stack<TreeNode<T>> stack = new Stack<>();
			while (node.right != null){
				stack.push(node);
				node = node.right;
			}
			if (!stack.isEmpty()) {
				stack.peek().right = node.left;
				node.left = root.left;
			}
			node.right = root.right;

			TreeNode<T> newRoot = node;
			while (!stack.isEmpty()){
				node = stack.pop();
				if (!stack.isEmpty())
					stack.peek().right = newRoot.assimilateOverlappingIntervals(node);
				else
					newRoot.left = newRoot.assimilateOverlappingIntervals(node);
			}
			return newRoot.balanceOut();
		}
	}

	
	static <T extends Comparable<? super T>> void rangeQueryLeft(TreeNode<T> node, Interval<T> query, Set<Interval<T>> result) {
		while (node != null) {
			if (query.contains(node.midpoint)) {
				result.addAll(node.increasing);
				if (node.right != null) {
					for (Interval<T> next : node.right)
						result.add(next);
				}
				node = node.left;
			} else {
				for (Interval<T> next: node.decreasing){
					if (next.isLeftOf(query))
						break;
					result.add(next);
				}
				node = node.right;
			}
		}
	}

	
	static <T extends Comparable<? super T>> void rangeQueryRight(TreeNode<T> node, Interval<T> query, Set<Interval<T>> result) {
		while (node != null) {
			if (query.contains(node.midpoint)) {
				result.addAll(node.increasing);
				if (node.left != null) {
					for (Interval<T> next : node.left)
						result.add(next);
				}
				node = node.right;
			} else {
				for (Interval<T> next: node.increasing){
					if (next.isRightOf(query))
						break;
					result.add(next);
				}
				node = node.left;
			}
		}
	}


	
	@Override
	public TreeNodeIterator iterator() {
		return new TreeNodeIterator();
	}

	class TreeNodeIterator implements Iterator<Interval<T>>{
		Stack<TreeNode<T>> stack = new Stack<>();
		TreeNode<T> subtreeRoot = TreeNode.this;
		TreeNode<T> currentNode;
		Interval<T> currentInterval;
		Iterator<Interval<T>> iterator = Collections.emptyIterator();

		@Override
		public boolean hasNext() {
			return subtreeRoot != null || !stack.isEmpty() || iterator.hasNext();
		}

		@Override
		public Interval<T> next() {
			if (!iterator.hasNext()) {
				while (subtreeRoot != null) {
					stack.push(subtreeRoot);
					subtreeRoot = subtreeRoot.left;
				}
				if (stack.isEmpty())
					throw new NoSuchElementException();
				currentNode = stack.pop();
				iterator = currentNode.increasing.iterator();
				subtreeRoot = currentNode.right;
			}
			currentInterval = iterator.next();
			return currentInterval;
		}

		@Override
		public void remove() {
			iterator.remove();
		}
	}
}
