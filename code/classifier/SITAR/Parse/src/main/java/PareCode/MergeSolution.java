package PareCode;

import org.antlr.v4.runtime.misc.Interval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


public class MergeSolution {

    public static void merge(List<Interval> intervals) {
        if (intervals.size() == 0 || intervals == null) {
        }
        Collections.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval interval1, Interval interval2) {
                return interval1.a - interval2.a;
            }
        });
        List<Interval> merged = new ArrayList<Interval>();
        for (int i = 0; i < intervals.size(); ++i) {
            int L = intervals.get(i).a, R = intervals.get(i).b;
            if (merged.size() == 0 || merged.get(merged.size() - 1).b < L-1) {
                merged.add(new Interval(L, R));
            } else {
                merged.get(merged.size() - 1).b = Math.max(merged.get(merged.size() - 1).b, R);
            }
        }
        intervals.clear();
        intervals.addAll(merged);
    }



