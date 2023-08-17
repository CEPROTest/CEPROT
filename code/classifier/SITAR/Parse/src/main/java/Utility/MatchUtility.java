package Utility;

import Action.CodeDiff;
import Action.GitAdapter;
import Date.DateAction;
import Obj.CommitMessage;
import Regrex.RegrexDefinations;
import Resource.Resource;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.diff.DiffEntry;
import org.eclipse.jgit.diff.DiffFormatter;
import org.eclipse.jgit.revwalk.RevCommit;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.ResourceBundle;

public class MatchUtility {

    public static CommitMessage findCorresponding(GitAdapter adapter,String file_path, List<CommitMessage> copyOfRange,boolean isIgnore) throws Exception {

        var Production_name=RegrexDefinations.testTransformProduct(file_path); 

        for(CommitMessage commitMessage:copyOfRange){
            if(commitMessage.getCommitId()==null||commitMessage.getLastCommitId()==null){
                continue;
            }
            Object[] diff_result = CodeDiff.diffCommitToCommit(adapter, commitMessage.getCommitId(), commitMessage.getLastCommitId()
            , isIgnore);
            List<DiffEntry> diffs=(List<DiffEntry>)diff_result[1];
            DiffFormatter formatter=(DiffFormatter)diff_result[0];
            var diff_fileNames=new ArrayList<String>();
            for(DiffEntry diffEntry:diffs){
                if(formatter.toFileHeader(diffEntry).toEditList().size()!=0){
                    if(diffEntry.getChangeType()== DiffEntry.ChangeType.DELETE){
                        diff_fileNames.add(diffEntry.getOldPath());
                    }else {
                        diff_fileNames.add(diffEntry.getNewPath());
                    }
                }
            }
            for(String fileName:diff_fileNames){
                if(fileName.equals(Production_name)){
                    return commitMessage;
                }
            }
        }
        Production_name=Production_name.toUpperCase();


        for(CommitMessage commitMessage:copyOfRange){

            if(commitMessage.getCommitId()==null||commitMessage.getLastCommitId()==null){
                continue;
            }
            Object diff_result[]=CodeDiff.diffCommitToCommit(adapter, commitMessage.getCommitId(), commitMessage.getLastCommitId()
                    , isIgnore);
            List<DiffEntry> diffs=(List<DiffEntry>)diff_result[1];
            DiffFormatter formatter=(DiffFormatter)diff_result[0];
            var diff_fileNames=new ArrayList<String>();
            for(DiffEntry diffEntry:diffs){
                if(formatter.toFileHeader(diffEntry).toEditList().size()!=0){
                    if(diffEntry.getChangeType()== DiffEntry.ChangeType.DELETE){
                        diff_fileNames.add(diffEntry.getOldPath().toUpperCase());
                    }else {
                        diff_fileNames.add(diffEntry.getNewPath().toUpperCase());
                    }
                }
            }
            for(String fileName:diff_fileNames){
                if(fileName.equals(Production_name.toUpperCase())){
                    return commitMessage;
                }
            }
        }
        return null;
    }





    public static List<CommitMessage> getCommits_range(int index,CommitMessage[] all){

        CommitMessage nowCommit=all[index];  
        List<CommitMessage> result=new ArrayList<>();
        for(int i=index;i<all.length;i++){
            if(isInRange(nowCommit,all[i])){
                result.add(all[i]);
            }
        }
        return result;
    }

    private static boolean isInRange(CommitMessage now, CommitMessage previous){
        ResourceBundle resourceBundle=ResourceBundle.getBundle("parameter");
        String PositiveTime=resourceBundle.getString("PositiveTime");
        return DateAction.get_diff(now.getCommitDate(), previous.getCommitDate()) <= Double.parseDouble(PositiveTime);

    }


    public static List<RevCommit> getCommits_range(GitAdapter adapter,CommitMessage index){
        try {
            ResourceBundle resourceBundle=ResourceBundle.getBundle("parameter");
            String PositiveTime=resourceBundle.getString("PositiveTime").trim();
            var endDate= new Date(DateAction.ConvertTDate(index.getCommitDate()).getTime()-Long.parseLong(PositiveTime)*1000);
            var files=adapter.getRevLog(index.getCommitId(), endDate);
            return files;
        } catch (ParseException e) {
            e.printStackTrace();
            return null;
        }
    }

  
    public static List<String> getFilesName(GitAdapter adapter, List<CommitMessage> commitMessages) {
        List<String> files_name = null;
        try {
            files_name = ProcedureUtility.getAllFiles(adapter, commitMessages, adapter.getProjectName());
        } catch (IOException e) {
            e.printStackTrace();
        }
        return files_name;
    }


    public static List<CommitMessage>  getCommitMessages(GitAdapter adapter) {
        var commitMessages=new ArrayList<CommitMessage>();
        File commitDirectory=new File(Resource.commitDirctory);
        if(!commitDirectory.exists()){
            commitDirectory.mkdir();
        }
        File Commits_File = new File(commitDirectory.getPath()+File.separator+ adapter.getProjectName() + ".csv");
        System.out.println(Commits_File.getPath());
        if (Commits_File.exists()) {
            commitMessages = (ArrayList<CommitMessage>) ProcedureUtility.getAllCommits(Commits_File);
        } else {
            try {
                commitMessages = (ArrayList<CommitMessage>) adapter.getNo_MergeCommitMessages();

                ProcedureUtility.WriteCommits(commitMessages, Commits_File);
            } catch (IOException | GitAPIException e) {
                e.printStackTrace();
            }
        }
        return commitMessages;
    }
}
