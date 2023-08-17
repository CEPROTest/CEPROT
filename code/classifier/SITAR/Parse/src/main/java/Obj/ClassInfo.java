package Obj;

import com.fasterxml.jackson.annotation.JsonIgnore;

import java.util.List;
import java.util.Objects;

public class ClassInfo {

    private String classFile;  

    private String className; 

    private String packages;

    private List<MethodInfo> methodInfos;


    private List<Integer> addLines;


    private List<Integer> delLines;

    private String type;


    private String path;


    @JsonIgnore
    private String classContent;


    private String SHA1;

    /**
     * 
     * @return
     */
    private String CommitID;


    private String ProjectName;


    private String pro_Commit;


    private String File_Date;

    private String Pro_Date;

    private String Pro_Path;

    private Boolean label;

    private Boolean predict;

    public Boolean getLabel() {
        return label;
    }

    public void setLabel(Boolean label) {
        this.label = label;
    }

    public Boolean getPredict() {
        return predict;
    }

    public void setPredict(Boolean predict) {
        this.predict = predict;
    }

    public String getPro_Path() {
        return Pro_Path;
    }

    public void setPro_Path(String pro_Path) {
        Pro_Path = pro_Path;
    }

    public String getFile_Date() {
        return File_Date;
    }

    public void setFile_Date(String file_Date) {
        File_Date = file_Date;
    }

    public String getPro_Date() {
        return Pro_Date;
    }

    public void setPro_Date(String pro_Date) {
        Pro_Date = pro_Date;
    }

    public String getPro_Commit() {
        return pro_Commit;
    }

    public void setPro_Commit(String pro_Commit) {
        this.pro_Commit = pro_Commit;
    }

    public String getProjectName() {
        return ProjectName;
    }

    public void setProjectName(String projectName) {
        ProjectName = projectName;
    }

    public String getCommitID() {
        return CommitID;
    }

    public void setCommitID(String commitID) {
        CommitID = commitID;
    }

    public String getSHA1() {
        return SHA1;
    }

    public void setSHA1(String SHA1) {
        this.SHA1 = SHA1;
    }







    public String getClassContent() {
        return classContent;
    }

    public void setClassContent(String classContent) {
        this.classContent = classContent;
    }

    public String getClassFile() {
        return classFile;
    }

    public void setClassFile(String classFile) {
        this.classFile = classFile;
    }

    public String getClassName() {
        return className;
    }

    public void setClassName(String className) {
        this.className = className;
    }

    public String getPackages() {
        return packages;
    }

    public void setPackages(String packages) {
        this.packages = packages;
    }

    public List<MethodInfo> getMethodInfos() {
        return methodInfos;
    }

    public void setMethodInfos(List<MethodInfo> methodInfos) {
        this.methodInfos = methodInfos;
    }

    public List<Integer> getAddLines() {
        return addLines;
    }

    public void setAddLines(List<Integer> addLines) {
        this.addLines = addLines;
    }

    public List<Integer> getDelLines() {
        return delLines;
    }

    public void setDelLines(List<Integer> delLines) {
        this.delLines = delLines;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    @Override
    public String toString() {
        return "ClassInfo{" +
                "classFile='" + classFile + '\'' +
                ", className='" + className + '\'' +
                ", packages='" + packages + '\'' +
                ", methodInfos=" + methodInfos +
                ", addLines=" + addLines +
                ", delLines=" + delLines +
                ", type='" + type + '\'' +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ClassInfo)) return false;
        ClassInfo classInfo = (ClassInfo) o;
        return Objects.equals(getClassFile(), classInfo.getClassFile()) && Objects.equals(getClassName(), classInfo.getClassName()) && Objects.equals(getPackages(), classInfo.getPackages()) && Objects.equals(getMethodInfos(), classInfo.getMethodInfos()) && Objects.equals(getAddLines(), classInfo.getAddLines()) && Objects.equals(getDelLines(), classInfo.getDelLines()) && Objects.equals(getType(), classInfo.getType()) && Objects.equals(getPath(), classInfo.getPath()) && Objects.equals(getClassContent(), classInfo.getClassContent()) && Objects.equals(getSHA1(), classInfo.getSHA1()) && Objects.equals(getCommitID(), classInfo.getCommitID()) && Objects.equals(getProjectName(), classInfo.getProjectName()) && Objects.equals(getPro_Commit(), classInfo.getPro_Commit()) && Objects.equals(getFile_Date(), classInfo.getFile_Date()) && Objects.equals(getPro_Date(), classInfo.getPro_Date());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getClassFile(), getClassName(), getPackages(), getMethodInfos(), getAddLines(), getDelLines(), getType(), getPath(), getClassContent(), getSHA1(), getCommitID(), getProjectName(), getPro_Commit(), getFile_Date(), getPro_Date());
    }
}