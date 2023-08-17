package SITAR;

import Action.CodeDiff;
import Action.GitAdapter;
import Bean.FeatureDo;
import Date.DateAction;
import IntervalTreeInstance.IntervalTree;
import Obj.ClassInfo;
import Obj.CommitMessage;
import PareCode.PareInstance;
import PareCode.SpecificTreeParser;
import Persistent.Serialization;
import Utility.gitInstance;
import org.apache.commons.io.FileUtils;
import org.eclipse.jgit.diff.DiffEntry;

import java.io.*;
import java.math.BigInteger;
import java.util.*;

import static Utility.MatchUtility.getCommitMessages;

public class MethodImplementation{
    public static void main(String[] args){
        File totalDir =new File("/SITAR/Method/");
        File[] categoriesDir = totalDir.listFiles();
        for(int i=0;i<categoriesDir.length;i++){
            System.out.println(categoriesDir[i].getName());
        }

        File FeatureDir=new File("/Feature");
        assert categoriesDir != null;
        System.out.println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        for(File file:categoriesDir) {
            // dir method
            File dir=new File(FeatureDir+File.separator+file.getName()+File.separator);
            System.out.println(file.getName());
            if(dir.exists()){
               continue;
            }
            try {
                var result=processCategory(file);
                Collections.sort(result, new Comparator<FeatureDo>() {
                    @Override
                    public int compare(FeatureDo o1, FeatureDo o2) {
                        if(DateAction.get_diff(o1.getProd_time(),o2.getProd_time())>0){
                            return 1;
                        }else  if(DateAction.get_diff(o1.getProd_time(),o2.getProd_time())==0){
                            return 0;
                        }else{
                            return -1;
                        }
                    }
                });
                String projectName=file.getName();
                dir=new File(FeatureDir+File.separator+projectName+File.separator);
                if(!dir.exists()){
                    FileUtils.forceMkdir(dir);
                }
                for(int i=0;i<result.size();i++){
                    try(BufferedWriter writer=new BufferedWriter(
                            new FileWriter(dir.getPath()+File.separator+(i+1)+".json")
                    )){
                        writer.write(Serialization.ObjToJSON(result.get(i)));
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static List<FeatureDo> processCategory(File file_dir) throws IOException {
        var filenames= FileUtils.listFiles(file_dir,null,true);
        String reposity=file_dir.getName();
        GitAdapter adapter = gitInstance.get(reposity, "master");
        adapter.initGit();
        List<CommitMessage> commitMessages = getCommitMessages(adapter);
        var result=new ArrayList<FeatureDo>();
        CommitMessage finalCommitMessage=commitMessages.get(0);


        for(File singleFile:filenames) {
            var classInfo = Serialization.json2Bean(readContents(singleFile), ClassInfo.class);

            FeatureDo featureDo = new FeatureDo();
            featureDo.setRepository(reposity);
            featureDo.setTest_path(classInfo.getPath());
            featureDo.setProd_path(classInfo.getPro_Path());
            featureDo.setTest_time(classInfo.getFile_Date());
            featureDo.setProd_time(classInfo.getPro_Date());
            featureDo.setPro_Commit(classInfo.getPro_Commit());
            featureDo.setTest_Commit(classInfo.getCommitID());
            double Hours_between;
            if(featureDo.getTest_time()==null || featureDo.getTest_time().equals("")){
                Hours_between = new BigInteger(String.valueOf(DateAction.get_diff(finalCommitMessage.getCommitDate(), featureDo.getProd_time()))).divide(new BigInteger(String.valueOf(60)))
                        .divide(new BigInteger(String.valueOf(60))).doubleValue();

            }else{
                Hours_between = new BigInteger(String.valueOf(DateAction.get_diff(featureDo.getTest_time(), featureDo.getProd_time()))).divide(new BigInteger(String.valueOf(60)))
                        .divide(new BigInteger(String.valueOf(60))).doubleValue();
            }
            if (Hours_between <= 48 && featureDo.getTest_time()!=null) {
                featureDo.setLabel("POSITIVE");
            } else if (Hours_between >= 480 && featureDo.getTest_time()!=null) {
                featureDo.setLabel("NEGATIVE");
            }else {
                featureDo.setLabel("UNDEFINE");
                continue;
            }

            int index=getIndex(featureDo.getPro_Commit(),commitMessages);
            if (index == -1){
                continue;
            }
            try{
                var proCommitMessage=commitMessages.get(index);

                var diffentry=adapter.getDiffOfFileInCommit(proCommitMessage, featureDo.getProd_path());
                if(diffentry.getChangeType()== DiffEntry.ChangeType.ADD){
                    String newContent=adapter.getCommitSpecificFileContent(proCommitMessage.getCommitId(),  featureDo.getProd_path());
                    var newIntervalTree= PareInstance.ParseFile(newContent);
                    fillFeatures(featureDo,newIntervalTree,"ADD");

                }else if(diffentry.getChangeType()==DiffEntry.ChangeType.DELETE){
                    String oldContent=adapter.getCommitSpecificFileContent(proCommitMessage.getLastCommitId(),  featureDo.getProd_path());
                    var oldIntervalTree= PareInstance.ParseFile(oldContent);
                    fillFeatures(featureDo,oldIntervalTree,"DEL");
                }else{
                    String newContent=adapter.getCommitSpecificFileContent(proCommitMessage.getCommitId(), diffentry.getNewPath());
                    String oldContent=adapter.getCommitSpecificFileContent(proCommitMessage.getLastCommitId(), diffentry.getOldPath());
                    var newIntervalTree= PareInstance.ParseFile(newContent);
                    var oldIntervalTree= PareInstance.ParseFile(oldContent);
                    var prodiff= CodeDiff.prepareDiffMethod(adapter,proCommitMessage.getCommitId(),
                            proCommitMessage.getLastCommitId(),CodeDiff.getDiffFormatter(true,null,adapter.getRepository()),diffentry,true);
                    
                    var addFeature=getModifyFeatures(newIntervalTree,prodiff,"ADD");
                    var delFeature=getModifyFeatures(oldIntervalTree,prodiff,"DEL");
                    fillModifyFeatures(featureDo,addFeature,"ADD");
                    fillModifyFeatures(featureDo,delFeature,"DEL");
                }
            }
            catch(NullPointerException e){
                System.out.println("NullPointerException caught, go on!");
                continue;
            }
            catch(Exception e){
                System.out.println("Exception caught, go on!");
                continue;
            }
            System.out.println(Serialization.ObjToJSON(featureDo));
            result.add(featureDo);
        }

        return result;
    }

    private static HashMap<SpecificTreeParser.ResultEnum, Integer> getModifyFeatures(HashMap<SpecificTreeParser.ResultEnum, IntervalTree<Integer>> IntervalTree, ClassInfo prodiff, String Category) {
        var result=new HashMap<SpecificTreeParser.ResultEnum, Integer>();
        switch (Category){
            case "ADD":
                var addLines=prodiff.getAddLines();
                for(Integer value:addLines){
                    IntervalTree.forEach((key, value1) -> {
                        result.putIfAbsent(key, 0);
                        if (value1.query(value).size() != 0) {
                            result.put(key, result.get(key) + 1);
                        }
                    });
                }
                break;
            case "DEL":
                var delLines=prodiff.getDelLines();
                for(Integer value:delLines){
                    IntervalTree.forEach((key, value1) -> {
                        result.putIfAbsent(key, 0);
                        if (value1.query(value).size() != 0) {
                            result.put(key, result.get(key) + 1);
                        }
                    });
                }
                break;
            default:
                break;
        }

        return result;
    }


    private static void fillModifyFeatures(FeatureDo featureDo, HashMap<SpecificTreeParser.ResultEnum, Integer> result, String category) {
        var Iterator=result.entrySet().iterator();
        switch (category){
            case "ADD":
                while(Iterator.hasNext()){
                    Map.Entry<SpecificTreeParser.ResultEnum,Integer> entry=Iterator.next();
                    SpecificTreeParser.ResultEnum resultEnum=entry.getKey();
                    int count=entry.getValue();
                    FillAddConcreteCount(featureDo, resultEnum, count);
                }
                break;
            case "DEL":
                while(Iterator.hasNext()){
                    Map.Entry<SpecificTreeParser.ResultEnum,Integer> entry=Iterator.next();
                    SpecificTreeParser.ResultEnum resultEnum=entry.getKey();
                    int count=entry.getValue();
                    FillDelConcreteCount(featureDo, resultEnum, count);
                }
                break;
            default:
                break;
        }

    }

    private static void FillDelConcreteCount(FeatureDo featureDo, SpecificTreeParser.ResultEnum resultEnum, int count) {
        switch (resultEnum){
            case Annotation:
                featureDo.setDel_annotation_line(count);
                break;
            case Expression:
                featureDo.setDel_condition_line(count);
                break;
            case PackageDeclaration:
                featureDo.setDel_packageid_line(count);
                break;
            case ImportDeclaration:
                featureDo.setDel_import_line(count);
                break;
            case NormalClassDeclaration:
                featureDo.setDel_classname_line(count);
                break;
            case FormalParameterList:
                featureDo.setDel_parameter_line(count);
                break;
            case MethodInvocation:
                featureDo.setDel_call_line(count);
                break;
            case ReturnStatement:
                featureDo.setDel_return_line(count);
                break;
            case FieldDeclaration:
                featureDo.setDel_field_line(count);
                break;
            default:
                break;
        }
    }

    private static void FillAddConcreteCount(FeatureDo featureDo, SpecificTreeParser.ResultEnum resultEnum, int count) {
        switch (resultEnum){
            case Annotation:
                featureDo.setAdd_annotation_line(count);
                break;
            case Expression:
                featureDo.setAdd_condition_line(count);
                break;
            case PackageDeclaration:
                featureDo.setAdd_packageid_line(count);
                break;
            case ImportDeclaration:
                featureDo.setAdd_import_line(count);
                break;
            case NormalClassDeclaration:
                featureDo.setAdd_classname_line(count);
                break;
            case FormalParameterList:
                featureDo.setAdd_parameter_line(count);
                break;
            case MethodInvocation:
                featureDo.setAdd_call_line(count);
                break;
            case ReturnStatement:
                featureDo.setAdd_return_line(count);
                break;
            case FieldDeclaration:
                featureDo.setAdd_field_line(count);
                break;
            default:
                break;
        }
    }

    private static void fillFeatures(FeatureDo featureDo, HashMap<SpecificTreeParser.ResultEnum, IntervalTree<Integer>> IntervalTree, String category) {
        var Iterator=IntervalTree.entrySet().iterator();
        switch (category){
            case "ADD":
                while(Iterator.hasNext()){
                    Map.Entry<SpecificTreeParser.ResultEnum,IntervalTree<Integer>> entry=Iterator.next();
                    SpecificTreeParser.ResultEnum resultEnum=entry.getKey();
                    IntervalTree tree=entry.getValue();
                    int count=getCountNumber(tree);
                    FillAddConcreteCount(featureDo, resultEnum, count);
                }
                break;
            case "DEL":
                while(Iterator.hasNext()){
                    Map.Entry<SpecificTreeParser.ResultEnum,IntervalTree<Integer>> entry=Iterator.next();
                    SpecificTreeParser.ResultEnum resultEnum=entry.getKey();
                    IntervalTree tree=entry.getValue();
                    int count=getCountNumber(tree);
                    FillDelConcreteCount(featureDo, resultEnum, count);
                }
                break;
            default:
                break;
        }

    }

    private static int getCountNumber(IntervalTree<Integer> tree) {
        return tree.stream().mapToInt(x->(x.getEnd()-x.getStart())+1).sum();
    }


    public static int getIndex(String commitID, List<CommitMessage> commitMessages){
        for(int i=0;i<commitMessages.size();i++){
            if(commitID.equals(commitMessages.get(i).getCommitId())){
                return i;
            }
        }
        return -1;
    }

    public static String readContents(File file){

        char[] contents=new char[(int)file.length()];
        try(FileReader reader=new FileReader(file)){
            reader.read(contents);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new String(contents);
    }
}