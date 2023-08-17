package Regrex;


import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class RegrexDefinations {
    private static String JavaCode_regex = "^.*.java$"; //The normal Java class
    private static String TestDirectory="^.*(/?src/test/).*.java$"; //The normal Test directory
    private static String ProducDirectory="^.*(/?src/main/).*.java$";
    private static String[] UnitTest1={"^Test.*.java$","^.*(Test).java$", "^.*(Tests).java$","^.*(TestCase).java$","^.*(TestCases).java$"};

    public static boolean isJava(String filename){
        return Pattern.matches(JavaCode_regex,filename);
    }


    public static boolean isTest(String filename){
        if(Pattern.matches(TestDirectory,filename)){
            var className=filename.substring(filename.lastIndexOf("/")+1); //Obtain the name of test code
            return isMathTest(className);
        }else {
            return false;
        }
    }


    private static  boolean isMathTest(String name){
        for(String regex:UnitTest1){
            if(Pattern.matches(regex,name)){
                return true;
            }
        }
        return false;
    }


    public static boolean isProduct(String filename){
        if(isJava(filename)){
            return !isTest(filename);
        }
        return false;
    }


    public static boolean isUnitTest(String file_path, List<String> files_name, String classContent) throws Exception {
        if(getTestType(file_path,files_name,classContent)== Type.UnitTest){
            return true;
        }else{
            return false;
        }
    }

    public static boolean isIntegrationTest(String file_path, List<String> files_name, String classContent) throws Exception {
        return !isUnitTest(file_path,files_name,classContent);
    }



    public static Type getTestType(String file_path, List<String> files_name, String classContent){

        if(!RegrexDefinations.isTest(file_path)){
            return null;
        }
        if(Pattern.matches("^.*(IntegrationTest.java)$", file_path)){
            return Type.IntegrationTest;
        }


        if((classContent!=null)&&(classContent.contains("@Tag(\"integration\")")||classContent.contains("@Category(IntegrationTest.class)"))){
            return Type.IntegrationTest;
        }
        if(is_findPro_Code(file_path,files_name)){ return Type.UnitTest; }
        else{ return Type.IntegrationTest; }
    }


    public static boolean is_findPro_Code(String file_path, List<String> files_name) {
        var Production_name = testTransformProduct(file_path);
        files_name = files_name.stream().sorted(Comparator.naturalOrder()).collect(Collectors.toList());  
        if (Collections.binarySearch(files_name, Production_name) >= 0) { 
            return true;
        }
        var Up_filesName = files_name.stream().map(String::toUpperCase).sorted(Comparator.naturalOrder()).collect(Collectors.toList());
        return Collections.binarySearch(Up_filesName, Production_name.toUpperCase()) >= 0;
    }

    public enum Type {
        UnitTest, IntegrationTest
    }

    public static String testTransformProduct(String testName){
        if(testName.matches(".*/Test.*\\.java")){
            return testName.replace("/Test", "/").replace("/test/", "/main/");
        }else{
            return testName.replaceAll("Test\\.java|Tests\\.java|TestCase\\.java|TestCases\\.java", ".java").replace("/test/", "/main/");
        }
    }


    public static String ProTransformTest(String proName, List<String> filesName) throws Exception {
        for(String fileName: filesName){
            if(isUnitTest(fileName,filesName,null)&&(testTransformProduct(fileName).equals(proName))){
                return fileName;
            }
        }
        return null;
    }

}
