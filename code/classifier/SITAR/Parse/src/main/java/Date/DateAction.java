package Date;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class DateAction {
	public static boolean CompareTo(String date1,String date2){
		SimpleDateFormat sdf  =new SimpleDateFormat("yyyy-MM-dd'T'HH-mm-ss");
       
        try {         
            Date dateD1 = sdf.parse(date1);
            Date dateD2 = sdf.parse(date2);
			return dateD1.getTime() >= dateD2.getTime();
        } catch (ParseException e1) {
            e1.printStackTrace();
        }
		return false;
	}
	public static long get_diff(String date1,String date2){

        try {         
            Date dateD1 = ConvertTDate(date1);
            Date dateD2 = ConvertTDate(date2);
            return (dateD1.getTime() - dateD2.getTime())/(long)1000; 
            
        } catch (ParseException e1) {
            e1.printStackTrace();
        }
		return -1;
	}

	public static Date ConvertTDate(String commitDate) throws ParseException {
		SimpleDateFormat sdf  =new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		return sdf.parse(commitDate);
	}
}
