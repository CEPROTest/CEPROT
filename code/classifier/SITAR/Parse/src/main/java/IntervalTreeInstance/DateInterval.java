package IntervalTreeInstance;

import java.util.Date;


public class DateInterval extends Interval<Date> {


	public DateInterval(){}


	public DateInterval(Date start, Date end, Bounded type){
		super(start, end, type);
	}


	public DateInterval(Date value, Unbounded type){
		super(value, type);
	}

	@Override
	protected Interval<Date> create() {
		return new DateInterval();
	}


	@Override
	public Date getMidpoint() {
		if (isEmpty())
			return null;
		long start = getStart() == null ? new Date(Long.MIN_VALUE).getTime() : getStart().getTime();
		long end = getEnd() == null ? new Date(Long.MAX_VALUE).getTime() : getEnd().getTime();

		if (start+1 == end && !isStartInclusive())
			return getEnd();
		if (start <= 0 && end >= 0)
			return new Date((end + start) / 2);

		return new Date(start + (end-start)/2);
	}
}
