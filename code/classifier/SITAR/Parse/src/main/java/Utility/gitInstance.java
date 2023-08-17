package Utility;

import Action.GitAdapter;

import java.util.ResourceBundle;


public class gitInstance {


    private static GitAdapter ReadInformation(String resourceName) {
        ResourceBundle res = ResourceBundle.getBundle(resourceName);
        String remotePath = res.getString("RemoteGit");
        String localPath = res.getString("LocalGit");
        String branchName = res.getString("branchName");
        String projectName = res.getString("projectName");

        String filePath = localPath + "/" + projectName; 

        return adapter = new GitAdapter(remotePath, filePath, branchName);
    }

    private static GitAdapter adapter = null;

    public static GitAdapter get(String resourceName) {

        return adapter = ReadInformation(resourceName);


    }

    public static GitAdapter get(String projectName, String branchName) {

        return adapter = ReadInformation(projectName, branchName);

    }

    private static GitAdapter ReadInformation(String projectName, String branchName) {

        var remotePath = "https://github.com/apache/" + projectName + ".git";
        var localPath = "java_data";


        String filePath = localPath + "/" + projectName;

        return adapter = new GitAdapter(remotePath, filePath, branchName);
    }
}
