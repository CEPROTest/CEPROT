package Bean;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

public class TestCode {
    @JsonProperty("prod_path")
    private String prodPath;
    @JsonProperty("prod_sha1")
    private String prodSha1;
    @JsonProperty("repository")
    private String repository;
    @JsonProperty("test_path")
    private String testPath;
    @JsonProperty("test_sha1")
    private String testSha1;
    @JsonProperty("commit_between")
    private Integer commitBetween;
    @JsonProperty("prod_qual")
    private String prodQual;
    @JsonProperty("prod_time")
    private ProdTimeDTO prodTime;
    @JsonProperty("prod_typ")
    private String prodTyp;
    @JsonProperty("sample_type")
    private String sampleType;
    @JsonProperty("test_qual")
    private String testQual;
    @JsonProperty("test_time")
    private TestTimeDTO testTime;
    @JsonProperty("test_typ")
    private String testTyp;
    @JsonProperty("add_annotation_line")
    private Integer addAnnotationLine;
    @JsonProperty("add_call_line")
    private Integer addCallLine;
    @JsonProperty("add_classname_line")
    private Integer addClassnameLine;
    @JsonProperty("add_condition_line")
    private Integer addConditionLine;
    @JsonProperty("add_field_line")
    private Integer addFieldLine;
    @JsonProperty("add_import_line")
    private Integer addImportLine;
    @JsonProperty("add_packageid_line")
    private Integer addPackageidLine;
    @JsonProperty("add_parameter_line")
    private Integer addParameterLine;
    @JsonProperty("add_return_line")
    private Integer addReturnLine;
    @JsonProperty("del_annotation_line")
    private Integer delAnnotationLine;
    @JsonProperty("del_call_line")
    private Integer delCallLine;
    @JsonProperty("del_classname_line")
    private Integer delClassnameLine;
    @JsonProperty("del_condition_line")
    private Integer delConditionLine;
    @JsonProperty("del_field_line")
    private Integer delFieldLine;
    @JsonProperty("del_import_line")
    private Integer delImportLine;
    @JsonProperty("del_packageid_line")
    private Integer delPackageidLine;
    @JsonProperty("del_parameter_line")
    private Integer delParameterLine;
    @JsonProperty("del_return_line")
    private Integer delReturnLine;

    private String true_label;

    public String getTrue_label() {
        return true_label;
    }

    public void setTrue_label(String true_label) {
        this.true_label = true_label;
    }

    @JsonIgnore
    public boolean isNull() {
        if(addAnnotationLine==0&&addCallLine==0&&addClassnameLine==0&&addFieldLine==0&&addConditionLine==0
        &&addImportLine==0&&addPackageidLine==0&&addReturnLine==0&&addParameterLine==0
        &&delAnnotationLine==0&&delCallLine==0&&delClassnameLine==0&&delFieldLine==0&&delConditionLine==0
                &&delImportLine==0&&delPackageidLine==0&&delReturnLine==0&&delParameterLine==0){
            return true;
        }
        return false;
    }

    @JsonIgnore
    public boolean isOther() {
        if(this.sampleType.equalsIgnoreCase("positive") ||
        this.sampleType.equalsIgnoreCase("negative")){
            return false;
        }
        return true;
    }

    public static class ProdTimeDTO {
        @JsonProperty("$date")
        public Long Date;
    }

    public static class TestTimeDTO {
        @JsonProperty("$date")
        public Long Date;
    }

    public String getProdPath() {
        return prodPath;
    }

    public void setProdPath(String prodPath) {
        this.prodPath = prodPath;
    }

    public String getProdSha1() {
        return prodSha1;
    }

    public void setProdSha1(String prodSha1) {
        this.prodSha1 = prodSha1;
    }

    public String getRepository() {
        return repository;
    }

    public void setRepository(String repository) {
        this.repository = repository;
    }

    public String getTestPath() {
        return testPath;
    }

    public void setTestPath(String testPath) {
        this.testPath = testPath;
    }

    public String getTestSha1() {
        return testSha1;
    }

    public void setTestSha1(String testSha1) {
        this.testSha1 = testSha1;
    }

    public Integer getCommitBetween() {
        return commitBetween;
    }

    public void setCommitBetween(Integer commitBetween) {
        this.commitBetween = commitBetween;
    }

    public String getProdQual() {
        return prodQual;
    }

    public void setProdQual(String prodQual) {
        this.prodQual = prodQual;
    }

    public ProdTimeDTO getProdTime() {
        return prodTime;
    }

    public void setProdTime(ProdTimeDTO prodTime) {
        this.prodTime = prodTime;
    }

    public String getProdTyp() {
        return prodTyp;
    }

    public void setProdTyp(String prodTyp) {
        this.prodTyp = prodTyp;
    }

    public String getSampleType() {
        return sampleType;
    }

    public void setSampleType(String sampleType) {
        this.sampleType = sampleType;
    }

    public String getTestQual() {
        return testQual;
    }

    public void setTestQual(String testQual) {
        this.testQual = testQual;
    }

    public TestTimeDTO getTestTime() {
        return testTime;
    }

    public void setTestTime(TestTimeDTO testTime) {
        this.testTime = testTime;
    }

    public String getTestTyp() {
        return testTyp;
    }

    public void setTestTyp(String testTyp) {
        this.testTyp = testTyp;
    }

    public Integer getAddAnnotationLine() {
        return addAnnotationLine;
    }

    public void setAddAnnotationLine(Integer addAnnotationLine) {
        this.addAnnotationLine = addAnnotationLine;
    }

    public Integer getAddCallLine() {
        return addCallLine;
    }

    public void setAddCallLine(Integer addCallLine) {
        this.addCallLine = addCallLine;
    }

    public Integer getAddClassnameLine() {
        return addClassnameLine;
    }

    public void setAddClassnameLine(Integer addClassnameLine) {
        this.addClassnameLine = addClassnameLine;
    }

    public Integer getAddConditionLine() {
        return addConditionLine;
    }

    public void setAddConditionLine(Integer addConditionLine) {
        this.addConditionLine = addConditionLine;
    }

    public Integer getAddFieldLine() {
        return addFieldLine;
    }

    public void setAddFieldLine(Integer addFieldLine) {
        this.addFieldLine = addFieldLine;
    }

    public Integer getAddImportLine() {
        return addImportLine;
    }

    public void setAddImportLine(Integer addImportLine) {
        this.addImportLine = addImportLine;
    }

    public Integer getAddPackageidLine() {
        return addPackageidLine;
    }

    public void setAddPackageidLine(Integer addPackageidLine) {
        this.addPackageidLine = addPackageidLine;
    }

    public Integer getAddParameterLine() {
        return addParameterLine;
    }

    public void setAddParameterLine(Integer addParameterLine) {
        this.addParameterLine = addParameterLine;
    }

    public Integer getAddReturnLine() {
        return addReturnLine;
    }

    public void setAddReturnLine(Integer addReturnLine) {
        this.addReturnLine = addReturnLine;
    }

    public Integer getDelAnnotationLine() {
        return delAnnotationLine;
    }

    public void setDelAnnotationLine(Integer delAnnotationLine) {
        this.delAnnotationLine = delAnnotationLine;
    }

    public Integer getDelCallLine() {
        return delCallLine;
    }

    public void setDelCallLine(Integer delCallLine) {
        this.delCallLine = delCallLine;
    }

    public Integer getDelClassnameLine() {
        return delClassnameLine;
    }

    public void setDelClassnameLine(Integer delClassnameLine) {
        this.delClassnameLine = delClassnameLine;
    }

    public Integer getDelConditionLine() {
        return delConditionLine;
    }

    public void setDelConditionLine(Integer delConditionLine) {
        this.delConditionLine = delConditionLine;
    }

    public Integer getDelFieldLine() {
        return delFieldLine;
    }

    public void setDelFieldLine(Integer delFieldLine) {
        this.delFieldLine = delFieldLine;
    }

    public Integer getDelImportLine() {
        return delImportLine;
    }

    public void setDelImportLine(Integer delImportLine) {
        this.delImportLine = delImportLine;
    }

    public Integer getDelPackageidLine() {
        return delPackageidLine;
    }

    public void setDelPackageidLine(Integer delPackageidLine) {
        this.delPackageidLine = delPackageidLine;
    }

    public Integer getDelParameterLine() {
        return delParameterLine;
    }

    public void setDelParameterLine(Integer delParameterLine) {
        this.delParameterLine = delParameterLine;
    }

    public Integer getDelReturnLine() {
        return delReturnLine;
    }

    public void setDelReturnLine(Integer delReturnLine) {
        this.delReturnLine = delReturnLine;
    }
}
