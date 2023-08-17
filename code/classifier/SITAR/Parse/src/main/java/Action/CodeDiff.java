package Action;

import Obj.ClassInfo;
import Obj.MethodInfo;
import com.google.common.base.Charsets;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.diff.*;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.patch.FileHeader;
import org.eclipse.jgit.treewalk.AbstractTreeIterator;
import org.eclipse.jgit.util.io.DisabledOutputStream;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class CodeDiff {


    public static List<ClassInfo> diffBranchToBranch(GitAdapter gitAdapter, String newBranchName, String oldBranchName) {
        return diffBranchToBranch(gitAdapter, newBranchName, oldBranchName, null);
    }


    public static List<ClassInfo> diffBranchToBranch(GitAdapter gitAdapter, String newBranchName, String oldBranchName, String commitId) {
        return diffMethods(gitAdapter, newBranchName, oldBranchName, commitId);
    }


    private static List<ClassInfo> diffMethods(GitAdapter gitAdapter, String newBranchName, String oldBranchName, String commitId) {
        try {
            Git git = gitAdapter.getGit();

            Ref newBranchRef = gitAdapter.checkOutAndPull(newBranchName);
            Ref oldBranchRef = gitAdapter.checkOutAndPull(oldBranchName);

            AbstractTreeIterator newTreeParser = gitAdapter.prepareTreeParser(newBranchRef.getObjectId().getName());

            if (null == commitId && oldBranchRef != null) {
                commitId = oldBranchRef.getObjectId().getName();
            }
            AbstractTreeIterator oldTreeParser = gitAdapter.prepareTreeParser(commitId);

            List<DiffEntry> diffs = git.diff().setOldTree(oldTreeParser).setNewTree(newTreeParser).setShowNameAndStatusOnly(true).call();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            DiffFormatter df = new DiffFormatter(out);
            df.setDiffComparator(RawTextComparator.WS_IGNORE_ALL);
            df.setRepository(git.getRepository());
            return batchPrepareDiffMethod(gitAdapter, newBranchName, oldBranchName, df, diffs, false);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new ArrayList<>();
    }

   

    public static List<ClassInfo> diff2Commit_concrete(GitAdapter gitAdapter, String newCommit, String oldCommit, boolean isIgnore) {
        try {
            Git git = gitAdapter.getGit();


            AbstractTreeIterator newTreeParser = gitAdapter.prepareTreeParser(newCommit);

            AbstractTreeIterator odlTreeParser = gitAdapter.prepareTreeParser(oldCommit);


            List<DiffEntry> diffs = git.diff().setOldTree(odlTreeParser).setNewTree(newTreeParser).setShowNameAndStatusOnly(true).call();


            ByteArrayOutputStream out = new ByteArrayOutputStream();

            DiffFormatter df = getDiffFormatter(isIgnore, out, git.getRepository());

            return batchPrepareDiffMethod(gitAdapter, newCommit, oldCommit, df, diffs, true);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new ArrayList<>();
    }


    public static List<DiffEntry>[] GetChanges_ByBlank(GitAdapter gitAdapter, String newCommit, String oldCommit) {
        try {

            Git git = gitAdapter.getGit();

            List<DiffEntry> diffs = gitAdapter.getFilesInRange(oldCommit, newCommit);


            var BlackIndex = diffs.stream().mapToInt(x -> {
                try {

                    FileHeader fileHeader = getDiffFormatter(true, DisabledOutputStream.INSTANCE, git.getRepository()).toFileHeader(x);

                    EditList editList = fileHeader.toEditList();   

                    if (editList.size() == 0) {
                        return 1;
                    } else {
                        return 0;
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                return 0;
            }).toArray();


            var BlackList = new ArrayList<DiffEntry>();
            var NonBlackList = new ArrayList<DiffEntry>();

            for (int i = 0; i < BlackIndex.length; i++) {
                if (BlackIndex[i] == 1) {    
                    BlackList.add(diffs.get(i));
                } else {
                    NonBlackList.add(diffs.get(i));
                }
            }

            return new List[]{NonBlackList, BlackList};

        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }


    public static List<ClassInfo> batchPrepareDiffMethod(final GitAdapter gitAdapter, final String newName, final String oldName, final DiffFormatter df, List<DiffEntry> diffs, Boolean isCommit) {
        int threadSize = 100;
        int dataSize = diffs.size();   
        int threadNum = dataSize / threadSize + 1;
        boolean special = dataSize % threadSize == 0;
        ExecutorService executorService = Executors.newFixedThreadPool(threadNum);

        List<Callable<List<ClassInfo>>> tasks = new ArrayList<>();
        Callable<List<ClassInfo>> task;  
        List<DiffEntry> cutList;
        for (int i = 0; i < threadNum; i++) {

            if (i == threadNum - 1) {
                if (special) {
                    break;
                }
                cutList = diffs.subList(threadSize * i, dataSize);
            } else {
                cutList = diffs.subList(threadSize * i, threadSize * (i + 1));
            }

            final List<DiffEntry> diffEntryList = cutList;
            task = () -> {
                List<ClassInfo> allList = new ArrayList<>();
                for (DiffEntry diffEntry : diffEntryList) {
                    ClassInfo classInfo;
                    if (isCommit) {
                        classInfo = prepareDiffMethod(gitAdapter, newName, oldName, df, diffEntry, true);
                    } else {
                        classInfo = prepareDiffMethod(gitAdapter, newName, oldName, df, diffEntry, false);
                    }

                    if (classInfo != null) {
                        classInfo.setCommitID(newName);
                        allList.add(classInfo);
                    }
                }
                return allList;
            };
            tasks.add(task);
        }
        List<ClassInfo> allClassInfoList = new ArrayList<>();
        try {
            List<Future<List<ClassInfo>>> results = executorService.invokeAll(tasks);
            for (Future<List<ClassInfo>> future : results) {
                allClassInfoList.addAll(future.get());
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            executorService.shutdown();
        }
        return allClassInfoList;
    }


    public static synchronized  ClassInfo prepareDiffMethod(GitAdapter gitAdapter, String newCommit, String oldCommit, DiffFormatter df, DiffEntry diffEntry, boolean isCommit) {
        List<MethodInfo> methodInfoList = new ArrayList<>();
        try {
            String newJavaPath = diffEntry.getNewPath();


            if (diffEntry.getChangeType() == DiffEntry.ChangeType.ADD) {  
                String newClassContent = GitAdapter.getClassContent(gitAdapter, newCommit, newJavaPath, isCommit);
                ASTGenerator newAstGenerator = new ASTGenerator(newClassContent);

                List<int[]> addLines = new ArrayList<>();
                List<int[]> delLines = new ArrayList<>();
                getChangeLines(df, diffEntry, addLines, delLines);

                return newAstGenerator.getClassInfo(gitAdapter.getProjectName(),translate(addLines), translate(delLines), DiffEntry.ChangeType.ADD.toString(), diffEntry.getNewPath(), newClassContent, diffEntry.getId(DiffEntry.Side.NEW));
            } else if (diffEntry.getChangeType() == DiffEntry.ChangeType.DELETE) {
                String oldJavaPath = diffEntry.getOldPath();
                String oldClassContent = GitAdapter.getClassContent(gitAdapter, oldCommit, oldJavaPath, isCommit);
                ASTGenerator oldAstGenerator = new ASTGenerator(oldClassContent);
                List<int[]> addLines = new ArrayList<>();
                List<int[]> delLines = new ArrayList<>();
                getChangeLines(df, diffEntry, addLines, delLines);

                return oldAstGenerator.getClassInfo(gitAdapter.getProjectName(),translate(addLines), translate(delLines), DiffEntry.ChangeType.DELETE.toString(), diffEntry.getOldPath(), oldClassContent, diffEntry.getId(DiffEntry.Side.OLD));
            }
            String newClassContent = GitAdapter.getClassContent(gitAdapter, newCommit, newJavaPath, isCommit);
            ASTGenerator newAstGenerator = new ASTGenerator(newClassContent);

            String oldJavaPath = diffEntry.getOldPath();

            String oldClassContent = GitAdapter.getClassContent(gitAdapter, oldCommit, oldJavaPath, isCommit);
            ASTGenerator oldAstGenerator = new ASTGenerator(oldClassContent);

            List<int[]> addLines = new ArrayList<>();
            List<int[]> delLines = new ArrayList<>();

            if (getChangeLines(df, diffEntry, addLines, delLines)) {
                return null;
            }

            MethodDeclaration[] newMethods = newAstGenerator.getMethods();
            MethodDeclaration[] oldMethods = oldAstGenerator.getMethods();
            Map<String, MethodDeclaration> methodsMap = new HashMap<>();
            for (MethodDeclaration oldMethod : oldMethods) {
                methodsMap.put(oldMethod.getName().toString() + oldMethod.parameters().toString(), oldMethod);
            }
            for (final MethodDeclaration method : newMethods) {
                if (!ASTGenerator.isMethodExist(method, methodsMap)) {
                    MethodInfo methodInfo = newAstGenerator.getMethodInfo(method);
                    methodInfoList.add(methodInfo);
                    continue;
                }
                if (!ASTGenerator.isMethodTheSame(method, methodsMap.get(method.getName().toString() + method.parameters().toString()))) {
                    MethodInfo methodInfo = newAstGenerator.getMethodInfo(method);
                    methodInfoList.add(methodInfo);
                }
            }
            return newAstGenerator.getClassInfo(gitAdapter.getProjectName(),methodInfoList, translate(addLines), translate(delLines), diffEntry.getChangeType().toString(), diffEntry.getNewPath(), newClassContent, diffEntry.getId(DiffEntry.Side.NEW));
        } catch (Exception e) {
            System.out.println(newCommit);
            e.printStackTrace();
        }
        return null;
    }

    public static boolean getChangeLines(DiffFormatter df, DiffEntry diffEntry, List<int[]> addLines, List<int[]> delLines) throws IOException {
        FileHeader fileHeader = df.toFileHeader(diffEntry);
        EditList editList = fileHeader.toEditList();   
        if (editList.size() == 0) {
            return true;
        }
            /*
             */
        for (Edit edit : editList) {
            if (edit.getLengthA() > 0) {
                delLines.add(new int[]{edit.getBeginA(), edit.getEndA()});
            }
            if (edit.getLengthB() > 0) {
                addLines.add(new int[]{edit.getBeginB(), edit.getEndB()});
            }
        }
        return false;
    }


    public static List<Integer> translate(List<int[]> listInt) {
        List<Integer> list = new ArrayList<>();
        for (int[] ints : listInt) {
            for (int i = ints[0]; i < ints[1]; i++) {
                list.add(i + 1);
            }
        }
        return list;
    }


    public static Object[] diffCommitToCommit(GitAdapter gitAdapter, String newCommit, String oldCommit, boolean isIgnore) throws IOException, GitAPIException {

        Git git = gitAdapter.getGit();

        AbstractTreeIterator oldTreeParser = gitAdapter.prepareTreeParser(oldCommit);

        AbstractTreeIterator newTreeParser = gitAdapter.prepareTreeParser(newCommit);

        List<DiffEntry> diffs = git.diff().setOldTree(oldTreeParser).setNewTree(newTreeParser).setShowNameAndStatusOnly(true).call();

        try (OutputStream tmpOutput = new ByteArrayOutputStream(2048)) {
            DiffFormatter formatter = getDiffFormatter(isIgnore, tmpOutput, gitAdapter.getRepository());
            formatter.format(diffs);

            return new Object[]{formatter, diffs};
        }
    }


    public static DiffFormatter getDiffFormatter(boolean isIgnore, OutputStream tmpOutput, Repository repository) {
        DiffFormatter formatter = new DiffFormatter(tmpOutput);
        formatter.setRepository(repository);
        if (isIgnore) {
            formatter.setDiffComparator(RawTextComparator.WS_IGNORE_ALL); 
        } else {
            formatter.setDiffComparator(RawTextComparator.DEFAULT);
        }
        formatter.setDiffAlgorithm(DiffAlgorithm.getAlgorithm(DiffAlgorithm.SupportedAlgorithm.MYERS));
        formatter.setDetectRenames(true); 
        formatter.getRenameDetector().setRenameScore(50); 
        return formatter;
    }

    public static String getDiff(File file1, File file2) {

        OutputStream out = new ByteArrayOutputStream();
        EditList diffList = null;
        try {
            RawText rt1 = new RawText(file1);
            RawText rt2 = new RawText(file2);
            diffList = new EditList();
            diffList.addAll(new HistogramDiff().diff(RawTextComparator.DEFAULT, rt1, rt2));
            new DiffFormatter(out).format(diffList, rt1, rt2);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return out.toString();
    }


    public static String getDiff(String beforeConent, String afterContent) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        RawText beforeRawText = new RawText(beforeConent.getBytes(Charsets.UTF_8));
        RawText afeterRawText = new RawText(afterContent.getBytes(Charsets.UTF_8));
        EditList diffList = new EditList();
        diffList.addAll(new HistogramDiff().diff(RawTextComparator.DEFAULT, beforeRawText, afeterRawText));
        new DiffFormatter(out).format(diffList, beforeRawText, afeterRawText);
        return out.toString(Charsets.UTF_8.name());
    }

}