package IntervalTreeInstance;

import java.util.Comparator;

public abstract class Interval<T extends Comparable<? super T>> {
	private T start, end;
	private boolean isStartInclusive, isEndInclusive;


	public enum Bounded {

		OPEN,

		CLOSED,

		CLOSED_RIGHT,

		CLOSED_LEFT
	}

	public enum Unbounded {

		OPEN_LEFT,


		CLOSED_LEFT,


		OPEN_RIGHT,


		CLOSED_RIGHT
	}


	public Interval(){
		isStartInclusive = true;
		isEndInclusive = true;
	}


	public Interval(T start, T end, Bounded type){
		this.start = start;
		this.end = end;
		if (type == null)
			type = Bounded.CLOSED;
		switch (type){
			case OPEN:
				break;
			case CLOSED:
				isStartInclusive = true;
				isEndInclusive = true;
				break;
			case CLOSED_RIGHT:
				isEndInclusive = true;
				break;
			default:
				isStartInclusive = true;
				break;
		}
	}


	public Interval(T value, Unbounded type){
		if (type == null)
			type = Unbounded.CLOSED_RIGHT;
		switch (type){
			case OPEN_LEFT:
				start = value;
				isStartInclusive = false;
				isEndInclusive = true;
				break;
			case CLOSED_LEFT:
				start = value;
				isStartInclusive = true;
				isEndInclusive = true;
				break;
			case OPEN_RIGHT:
				end = value;
				isStartInclusive = true;
				isEndInclusive = false;
				break;
			default:
				end = value;
				isStartInclusive = true;
				isEndInclusive = true;
				break;
		}
	}


	public boolean isEmpty() {
		if (start == null || end == null)
			return false;
		int compare = start.compareTo(end);
		if (compare>0)
			return true;
		if (compare == 0 && (!isEndInclusive || !isStartInclusive))
			return true;
		return false;
	}

	protected abstract Interval<T> create();


	public abstract T getMidpoint();


	protected Interval<T> create(T start, boolean isStartInclusive, T end, boolean isEndInclusive){
		Interval<T> interval = create();
		interval.start = start;
		interval.isStartInclusive = isStartInclusive;
		interval.end = end;
		interval.isEndInclusive = isEndInclusive;
		return interval;
	}

	/** Returns the start point of the interval. */
	public T getStart(){
		return start;
	}
	/** Returns the end point of the interval. */
	public T getEnd(){
		return end;
	}
	/** Returns {@code true}, if the start point is a part of the interval, or false otherwise. */
	public boolean isStartInclusive(){
		return isStartInclusive;
	}
	/** Returns {@code true}, if the end point is a part of the interval, or false otherwise. */
	public boolean isEndInclusive(){
		return isEndInclusive;
	}

	/**
	 * Determines if the current interval is a single point.
	 *
	 * @return {@code true}, if the current interval represents a single point.
	 */
	public boolean isPoint(){
		if (start == null || end == null) {
			return false;
		}
		return start.compareTo(end) == 0 && isStartInclusive && isEndInclusive;
	}

	/**
	 * Determines if the current interval contains a query point.
	 *
	 * @param query The point.
	 * @return {@code true}, if the current interval contains the {@code query} point or false otherwise.
	 */
	public boolean contains(T query){
		if (isEmpty() || query == null) {
			return false;
		}

		int startCompare = start == null ? 1 : query.compareTo(start);
		int endCompare = end == null ? -1 : query.compareTo(end);
		if (startCompare > 0 && endCompare < 0) {
			return true;
		}
		return (startCompare == 0 && isStartInclusive) || (endCompare == 0 && isEndInclusive);
	}

	/**
	 * Returns an interval, representing the intersection of two intervals. More formally, for every
	 * point {@code x} in the returned interval, {@code x} will belong in both the current interval
	 * and the {@code other} interval.
	 *
	 * @param other The other interval
	 * @return The intersection of the current interval wih the {@code other} interval.
	 */
	public Interval<T> getIntersection(Interval<T> other){
		if (other == null || isEmpty() || other.isEmpty())
			return null;
		if ((other.start == null && start != null) || (start != null && start.compareTo(other.start)>0))
			return other.getIntersection(this);
		if (end != null && other.start != null && (end.compareTo(other.start) < 0 || (end.compareTo(other.start) == 0 && (!isEndInclusive || !other.isStartInclusive))))
			return null;

		T newStart, newEnd;
		boolean isNewStartInclusive, isNewEndInclusive;

		if (other.start == null){
			newStart = null;
			isNewStartInclusive = true;
		} else {
			newStart = other.start;
			if (start != null && other.start.compareTo(start) == 0)
				isNewStartInclusive = other.isStartInclusive && isStartInclusive;
			else
				isNewStartInclusive = other.isStartInclusive;
		}

		if (end == null){
			newEnd = other.end;
			isNewEndInclusive = other.isEndInclusive;
		} else if (other.end == null){
			newEnd = end;
			isNewEndInclusive = isEndInclusive;
		} else {
			int compare = end.compareTo(other.end);
			if (compare == 0){
				newEnd = end;
				isNewEndInclusive = isEndInclusive && other.isEndInclusive;
			} else if (compare < 0){
				newEnd = end;
				isNewEndInclusive = isEndInclusive;
			} else {
				newEnd = other.end;
				isNewEndInclusive = other.isEndInclusive;
			}
		}
		Interval<T> intersection = create(newStart, isNewStartInclusive, newEnd, isNewEndInclusive);
		return intersection.isEmpty() ? null : intersection;
	}


	public boolean contains(Interval<T> another){
		if (another == null || isEmpty() || another.isEmpty()){
			return false;
		}
		Interval<T> intersection = getIntersection(another);
		return intersection != null && intersection.equals(another);
	}


	public boolean intersects(Interval<T> query){
		if (query == null)
			return false;
		Interval<T> intersection = getIntersection(query);
		return intersection != null;
	}


	public boolean isRightOf(T point, boolean inclusive){
		if (point == null || start == null)
			return false;
		int compare = point.compareTo(start);
		if (compare != 0)
			return compare < 0;
		return !isStartInclusive() || !inclusive;
	}


	public boolean isRightOf(T point){
		return isRightOf(point, true);
	}


	public boolean isRightOf(Interval<T> other){
		if (other == null || other.isEmpty())
			return false;
		return isRightOf(other.end, other.isEndInclusive());
	}

	public boolean isLeftOf(T point, boolean inclusive){
		if (point == null || end == null)
			return false;
		int compare = point.compareTo(end);
		if (compare != 0)
			return compare > 0;
		return !isEndInclusive() || !inclusive;
	}

	
	public boolean isLeftOf(T point){
		return isLeftOf(point, true);
	}


	public boolean isLeftOf(Interval<T> other){
		if (other == null || other.isEmpty())
			return false;
		return isLeftOf(other.start, other.isStartInclusive());
	}


	private int compareStarts(Interval<T> other){
		if (start == null && other.start == null)
			return 0;
		if (start == null)
			return -1;
		if (other.start == null)
			return 1;
		int compare = start.compareTo(other.start);
		if (compare != 0)
			return compare;
		if (isStartInclusive ^ other.isStartInclusive)
			return isStartInclusive ? -1 : 1;
		return 0;
	}


	private int compareEnds(Interval<T> other){
		if (end == null && other.end == null)
			return 0;
		if (end == null)
			return 1;
		if (other.end == null)
			return -1;
		int compare = end.compareTo(other.end);
		if (compare != 0)
			return compare;
		if (isEndInclusive ^ other.isEndInclusive)
			return isEndInclusive ? 1 : -1;
		return 0;
	}

	
	public static Comparator<Interval> sweepLeftToRight = new Comparator<Interval>() {
		@Override
		public int compare(Interval a, Interval b) {
			int compare = a.compareStarts(b);
			if (compare != 0)
				return compare;
			compare = a.compareEnds(b);
			if (compare != 0)
				return compare;
			return a.compareSpecialization(b);
		}
	};

	
	public static Comparator<Interval> sweepRightToLeft = new Comparator<Interval>() {
		@Override
		public int compare(Interval a, Interval b) {
			int compare = b.compareEnds(a);
			if (compare != 0)
				return compare;
			compare = b.compareStarts(a);
			if (compare != 0)
				return compare;
			return a.compareSpecialization(b);
		}
	};

	
	protected int compareSpecialization(Interval<T> other){
		return 0;
	}

	@Override
	public int hashCode() {
		int prime = 31;
		int result = start == null ? 0 : start.hashCode();
		result = prime * result +(end == null ? 0 : end.hashCode());
		result = prime * result + (isStartInclusive ? 1 : 0);
		result = prime * result + (isEndInclusive ? 1 : 0);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj == null || !(obj instanceof Interval))
			return false;
		Interval<T> other = (Interval<T>) obj;
		if (start == null ^ other.start == null)
			return false;
		if (end == null ^ other.end == null)
			return false;
		if (isEndInclusive ^ other.isEndInclusive)
			return false;
		if (isStartInclusive ^ other.isStartInclusive)
			return false;
		if (start != null && !start.equals(other.start))
			return false;
		if (end != null && !end.equals(other.end))
			return false;
		return true;
	}


	public Builder builder(){
		return new Builder(this);
	}

	
	public class Builder {
		private Interval<T> interval;

		
		private Builder(Interval<T> ref){
			interval = ref.create();
		}

		
		public Builder greater(T start){
			interval.start = start;
			interval.isStartInclusive = false;
			return this;
		}

		
		public Builder greaterEqual(T start){
			interval.start = start;
			interval.isStartInclusive = true;
			return this;
		}

		
		public Builder less(T end){
			interval.end = end;
			interval.isEndInclusive = false;
			return this;
		}

		
		public Builder lessEqual(T end){
			interval.end = end;
			interval.isEndInclusive = true;
			return this;
		}

		
		public Interval<T> build(){
			return interval;
		}
	}
}
