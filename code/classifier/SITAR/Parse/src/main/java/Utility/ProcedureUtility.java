package Utility;

import Action.CodeDiff;
import Action.GitAdapter;
import Obj.ClassInfo;
import Obj.CommitMessage;
import Persistent.Serialization;
import Resource.Resource;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.diff.DiffEntry;
import org.eclipse.jgit.diff.DiffFormatter;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;


public class ProcedureUtility {

    public static List<CommitMessage> getAllCommits(File commits_file) {
        var list=new ArrayList<CommitMessage>();
        try {
            var reader=new BufferedReader(new FileReader(commits_file));
            String line;
            while((line=reader.readLine())!=null){
                CommitMessage commitMessage= Serialization.json2Bean(line, CommitMessage.class);
                list.add(commitMessage);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return list;
    }



    public static ArrayList<String> getAllFiles(GitAdapter gitAdapter, List<CommitMessage> commitments, String projectName) throws IOException {
        File fileDirectory=new File(Resource.allFilesDirctory);
        if(!fileDirectory.exists()){
            fileDirectory.mkdir();
        }
        System.out.println(fileDirectory.getAbsoluteFile());
        File file =new File(fileDirectory.getPath()+File.separator+projectName+".csv");
        var uniqueList=new ArrayList<String>();
        var uniqueSet=new HashSet<String>();
        if(file.exists()){
            var reader=new BufferedReader(new FileReader(file));
            String line;
            while((line=reader.readLine())!=null){
                uniqueList.add(line);
            }
        }else{
            for(CommitMessage commitMessage:commitments){
                var list=gitAdapter.getJavaFilesCommit(commitMessage.getCommitId());
                for(var fileName:list){
                    uniqueSet.add(fileName);
                }
            }
            var writer=new BufferedWriter(new FileWriter(file));
            uniqueSet.forEach((file_name)-> {
                try {
                    uniqueList.add(file_name);
                    writer.write(file_name+"\r\n");
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
            writer.close();
        }
        return uniqueList;
    }


    private static void ProcessCommits(GitAdapter gitAdapter, List<CommitMessage> commitments) {
        File pFile=new File("./"+gitAdapter.getProjectName()+"_Diffs");
        try {
            FileUtils.deleteDirectory(pFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        pFile.mkdirs();
        File uFile=new File("./"+gitAdapter.getProjectName()+"_NoDiffs.csv");
        if(uFile.exists()){uFile.delete();}
        try {
            uFile.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        SimpleDateFormat dateFormat=new SimpleDateFormat("yyyy-MM-dd HH.mm.ss");
        for(CommitMessage message:commitments){
            if(message.getLastCommitId()==null){
                continue;
            }
            try {
                List<DiffEntry> diffs= (List<DiffEntry>) CodeDiff.diffCommitToCommit(gitAdapter,message.getCommitId(),message.getLastCommitId(),false)[1];
                List<DiffEntry> testDiff=new ArrayList<>();
                boolean isHaveTest=false;
                for(DiffEntry entry:diffs){
                    if(entry.getNewPath().contains(".java")){
                        testDiff.add(entry);
                    }
                    if(entry.getNewPath().contains("Test")){
                        isHaveTest=true;
                    }
                }
                if(isHaveTest){
                    File file=new File(pFile.getName()+File.separator+dateFormat.format(message.getDate())+".csv");
                    BufferedWriter writer=new BufferedWriter(new FileWriter(file));
                    for(DiffEntry entry:testDiff){
                        writer.write(message.getCommitId()+" "+entry.getChangeType()+" "+entry.getNewPath()+"\r\n");
                    }
                    writer.close();
                }else{
                    BufferedWriter writer=new BufferedWriter(new FileWriter(uFile, true));
                    writer.write(message.getCommitId()+"\r\n");
                    writer.close();
                }
            } catch (IOException | GitAPIException | ParseException e) {
                e.printStackTrace();
            }
        }
    }


    public static void WriteCommits(List<CommitMessage> commitments, File file) {
        try {
            if(!file.exists()){
                file.createNewFile();
            }else{
                return ;
            }
            BufferedWriter writer=new BufferedWriter(new FileWriter(file));
            for(CommitMessage commitmessage:commitments){
                writer.write(Serialization.ObjToJSON(commitmessage)+"\r\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static List<ClassInfo> match(GitAdapter adapter, CommitMessage commitment,boolean isIgnore) throws Exception {
        if(commitment.getCommitId()==null|| commitment.getLastCommitId()==null){
               return null;
            }

        return CodeDiff.diff2Commit_concrete(adapter, commitment.getCommitId(), commitment.getLastCommitId(),isIgnore);

    }


    public static List<ImmutablePair<DiffEntry,String>> getCodesByFilter(GitAdapter adapter, List<CommitMessage> commitments, boolean isIgnore, String regex) throws Exception {
        var result = new ArrayList<ImmutablePair<DiffEntry, String>>();

        for (CommitMessage commitMessage : commitments) {

            if (commitMessage.getCommitId() == null || commitMessage.getLastCommitId() == null) {
                continue;
            }

            Object[] diff_result = CodeDiff.diffCommitToCommit(adapter,commitMessage.getCommitId(), commitMessage.getLastCommitId(), isIgnore);
            List<DiffEntry> diffs = (List<DiffEntry>) diff_result[1];
            DiffFormatter formatter = (DiffFormatter) diff_result[0];
            for (DiffEntry diffEntry : diffs) {
                if (formatter.toFileHeader(diffEntry).toEditList().size() != 0) {
                    if (diffEntry.getChangeType() == DiffEntry.ChangeType.DELETE && Pattern.matches(regex, diffEntry.getOldPath())) {
                        var content = adapter.getCommitSpecificFileContent(commitMessage.getLastCommitId(), diffEntry.getOldPath());
                        result.add(new ImmutablePair<>(diffEntry, content));
                    } else if (Pattern.matches(regex, diffEntry.getNewPath())) {
                        var content = adapter.getCommitSpecificFileContent(commitMessage.getCommitId(), diffEntry.getNewPath());
                        result.add(new ImmutablePair<>(diffEntry, content));
                    }
                }
            }
        }
        return result;
    }

    
    public static void WriteClassInfo(File file,ClassInfo testClass, ArrayList<ClassInfo> production_list) throws IOException {
        if(file.exists()){
            file =new File((file.getPath().substring(0,file.getPath().indexOf(".json"))+"_"+(new Random().nextInt(100000)+".json")));
        }
        BufferedWriter writer=new BufferedWriter(new FileWriter(file,false));
        writer.write(Serialization.ObjToJSON(testClass)+"\r\n");
        if(production_list!=null){
            for(ClassInfo info:production_list){
                writer.write(Serialization.ObjToJSON(info)+"\r\n");
            }
        }
        writer.close();
    }
}
