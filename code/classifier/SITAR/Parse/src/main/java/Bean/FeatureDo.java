package Bean;

import com.fasterxml.jackson.annotation.JsonIgnore;

public class FeatureDo {

    @JsonIgnore
    String filename;

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    String repository;
    String prod_path;
    String test_path;
    String prod_time;
    String test_time;
    @JsonIgnore
    String test_type;
    int add_annotation_line=0;
    int add_call_line=0;
    int add_classname_line=0;
    int add_condition_line=0;
    int add_field_line=0;
    int add_import_line=0;
    int add_packageid_line=0;
    int add_parameter_line=0;
    int add_return_line=0;
    int del_annotation_line=0;
    int del_call_line=0;
    int del_classname_line=0;
    int del_condition_line=0;
    int del_field_line=0;
    int del_import_line=0;
    int del_packageid_line=0;
    int del_parameter_line=0;
    int del_return_line=0;
    String label;

    String pro_Commit;
    String test_Commit;

    String true_label;

    public String getTrue_label() {
        return true_label;
    }

    public void setTrue_label(String true_label) {
        this.true_label = true_label;
    }

    public String getPro_Commit() {
        return pro_Commit;
    }

    public void setPro_Commit(String pro_Commit) {
        this.pro_Commit = pro_Commit;
    }

    public String getTest_Commit() {
        return test_Commit;
    }

    public void setTest_Commit(String test_Commit) {
        this.test_Commit = test_Commit;
    }

    public String getRepository() {
        return repository;
    }

    public void setRepository(String repository) {
        this.repository = repository;
    }

    public String getProd_path() {
        return prod_path;
    }

    public void setProd_path(String prod_path) {
        this.prod_path = prod_path;
    }

    public String getTest_path() {
        return test_path;
    }

    public void setTest_path(String test_path) {
        this.test_path = test_path;
    }

    public String getProd_time() {
        return prod_time;
    }

    public void setProd_time(String prod_time) {
        this.prod_time = prod_time;
    }

    public String getTest_time() {
        return test_time;
    }

    public void setTest_time(String test_time) {
        this.test_time = test_time;
    }

    public String getTest_type() {
        return test_type;
    }

    public void setTest_type(String test_type) {
        this.test_type = test_type;
    }

    public int getAdd_annotation_line() {
        return add_annotation_line;
    }

    public void setAdd_annotation_line(int add_annotation_line) {
        this.add_annotation_line = add_annotation_line;
    }

    public int getAdd_call_line() {
        return add_call_line;
    }

    public void setAdd_call_line(int add_call_line) {
        this.add_call_line = add_call_line;
    }

    public int getAdd_classname_line() {
        return add_classname_line;
    }

    public void setAdd_classname_line(int add_classname_line) {
        this.add_classname_line = add_classname_line;
    }

    public int getAdd_condition_line() {
        return add_condition_line;
    }

    public void setAdd_condition_line(int add_condition_line) {
        this.add_condition_line = add_condition_line;
    }

    public int getAdd_field_line() {
        return add_field_line;
    }

    public void setAdd_field_line(int add_field_line) {
        this.add_field_line = add_field_line;
    }

    public int getAdd_import_line() {
        return add_import_line;
    }

    public void setAdd_import_line(int add_import_line) {
        this.add_import_line = add_import_line;
    }

    public int getAdd_packageid_line() {
        return add_packageid_line;
    }

    public void setAdd_packageid_line(int add_packageid_line) {
        this.add_packageid_line = add_packageid_line;
    }

    public int getAdd_parameter_line() {
        return add_parameter_line;
    }

    public void setAdd_parameter_line(int add_parameter_line) {
        this.add_parameter_line = add_parameter_line;
    }

    public int getAdd_return_line() {
        return add_return_line;
    }

    public void setAdd_return_line(int add_return_line) {
        this.add_return_line = add_return_line;
    }

    public int getDel_annotation_line() {
        return del_annotation_line;
    }

    public void setDel_annotation_line(int del_annotation_line) {
        this.del_annotation_line = del_annotation_line;
    }

    public int getDel_call_line() {
        return del_call_line;
    }

    public void setDel_call_line(int del_call_line) {
        this.del_call_line = del_call_line;
    }

    public int getDel_classname_line() {
        return del_classname_line;
    }

    public void setDel_classname_line(int del_classname_line) {
        this.del_classname_line = del_classname_line;
    }

    public int getDel_condition_line() {
        return del_condition_line;
    }

    public void setDel_condition_line(int del_condition_line) {
        this.del_condition_line = del_condition_line;
    }

    public int getDel_field_line() {
        return del_field_line;
    }

    public void setDel_field_line(int del_field_line) {
        this.del_field_line = del_field_line;
    }

    public int getDel_import_line() {
        return del_import_line;
    }

    public void setDel_import_line(int del_import_line) {
        this.del_import_line = del_import_line;
    }

    public int getDel_packageid_line() {
        return del_packageid_line;
    }

    public void setDel_packageid_line(int del_packageid_line) {
        this.del_packageid_line = del_packageid_line;
    }

    public int getDel_parameter_line() {
        return del_parameter_line;
    }

    public void setDel_parameter_line(int del_parameter_line) {
        this.del_parameter_line = del_parameter_line;
    }

    public int getDel_return_line() {
        return del_return_line;
    }

    public void setDel_return_line(int del_return_line) {
        this.del_return_line = del_return_line;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "FeatureDo{" +
                "repository='" + repository + '\'' +
                ", prod_path='" + prod_path + '\'' +
                ", test_path='" + test_path + '\'' +
                ", prod_time='" + prod_time + '\'' +
                ", test_time='" + test_time + '\'' +
                ", test_type='" + test_type + '\'' +
                ", add_annotation_line=" + add_annotation_line +
                ", add_call_line=" + add_call_line +
                ", add_classname_line=" + add_classname_line +
                ", add_condition_line=" + add_condition_line +
                ", add_field_line=" + add_field_line +
                ", add_import_line=" + add_import_line +
                ", add_packageid_line=" + add_packageid_line +
                ", add_parameter_line=" + add_parameter_line +
                ", add_return_line=" + add_return_line +
                ", del_annotation_line=" + del_annotation_line +
                ", del_call_line=" + del_call_line +
                ", del_classname_line=" + del_classname_line +
                ", del_condition_line=" + del_condition_line +
                ", del_field_line=" + del_field_line +
                ", del_import_line=" + del_import_line +
                ", del_packageid_line=" + del_packageid_line +
                ", del_parameter_line=" + del_parameter_line +
                ", del_return_line=" + del_return_line +
                ", label='" + label + '\'' +
                ", pro_Commit='" + pro_Commit + '\'' +
                ", test_Commit='" + test_Commit + '\'' +
                '}';
    }

    @JsonIgnore
    public boolean isEmpty() {
        if((add_annotation_line==0)&&
                (add_call_line==0)&&
        add_classname_line==00&&
        add_condition_line==0&&
        add_field_line==0&&
        add_import_line==0&&
        add_packageid_line==0&&
        add_parameter_line==0&&
        add_return_line==0&&
        del_annotation_line==0&&
        del_call_line==0&&
        del_classname_line==0&&
        del_condition_line==0&&
        del_field_line==0&&
        del_import_line==0&&
        del_packageid_line==0&&
        del_parameter_line==0&&
        del_return_line==0){
            return true;
        }
        return false;
    }
}