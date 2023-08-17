package Action;

import Obj.ClassInfo;
import Obj.MethodInfo;
import Regrex.RegrexDefinations;
import org.apache.commons.codec.binary.Base64;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.*;
import org.eclipse.jgit.lib.AbbreviatedObjectId;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Demo class
 *
 */
public class ASTGenerator {
    private final String javaText;
    private CompilationUnit compilationUnit;
    public ASTGenerator(String javaText) {
        this.javaText = javaText;
        this.initCompilationUnit();
    }

    private void initCompilationUnit() {
        final ASTParser astParser = ASTParser.newParser(8);
        final Map<String, String> options = JavaCore.getOptions();
        JavaCore.setComplianceOptions(JavaCore.VERSION_1_8, options);
        astParser.setCompilerOptions(options);
        astParser.setKind(ASTParser.K_COMPILATION_UNIT);
        astParser.setResolveBindings(true);
        astParser.setBindingsRecovery(true);
        astParser.setStatementsRecovery(true);
        astParser.setSource(javaText.toCharArray());
        compilationUnit = (CompilationUnit) astParser.createAST(null);
    }

    /**
     */
    public String getPackageName() {
        if (compilationUnit == null) {
            return "";
        }
        PackageDeclaration packageDeclaration = compilationUnit.getPackage();
        if (packageDeclaration == null){
            return "";
        }
        return packageDeclaration.getName().toString();
    }

    /**
     */
    public TypeDeclaration getJavaClass() {
        if (compilationUnit == null) {
            return null;
        }
        TypeDeclaration typeDeclaration = null;
        final List<?> types = compilationUnit.types();
        for (final Object type : types) {
            if (type instanceof TypeDeclaration) {
                typeDeclaration = (TypeDeclaration) type;
                break;
            }
        }
        return typeDeclaration;
    }

    public MethodDeclaration[] getMethods() {
        TypeDeclaration typeDec = getJavaClass();
        if (typeDec == null) {
            return new MethodDeclaration[]{};
        }
        return typeDec.getMethods();
    }

    public List<MethodInfo> getMethodInfoList() {
        MethodDeclaration[] methodDeclarations = getMethods();
        List<MethodInfo> methodInfoList = new ArrayList<>();
        for (MethodDeclaration method: methodDeclarations) {
            MethodInfo methodInfo = new MethodInfo();
            setMethodInfo(methodInfo, method);
            methodInfoList.add(methodInfo);
        }
        return methodInfoList;
    }


    public ClassInfo getClassInfo(String projectName,List<MethodInfo> methodInfos, List<Integer> addLines, List<Integer> delLines, String actionName, String path, String newClassContent, AbbreviatedObjectId id) {
        TypeDeclaration typeDec = getJavaClass();

        if(!RegrexDefinations.isJava(path)){
            return null;
        }

        if (typeDec == null) {
            return SetClassInfo(projectName,null,null,methodInfos,
                    addLines, delLines, actionName, path, newClassContent, id);
        }

        return SetClassInfo(projectName,getJavaClass().getName().toString(),getJavaClass().getName().toString(),methodInfos,
                addLines, delLines, actionName, path, newClassContent, id);
    }


       private ClassInfo SetClassInfo(String projectName,String ClassFile,String ClassName,List<MethodInfo> methodInfos, List<Integer> addLines, List<Integer> delLines, String actionName, String path, String newClassContent, AbbreviatedObjectId id) {
        ClassInfo classInfo = new ClassInfo();
        classInfo.setClassFile(ClassFile);
        classInfo.setClassName(ClassName);
        classInfo.setPackages(getPackageName());
        classInfo.setMethodInfos(methodInfos);
        classInfo.setAddLines(addLines);
        classInfo.setDelLines(delLines);
        classInfo.setType(actionName);
        classInfo.setPath(path);
        classInfo.setClassContent(newClassContent);
        classInfo.setSHA1(id.name());
        classInfo.setProjectName(projectName);

        return classInfo;
    }


    public ClassInfo getClassInfo(String projectName,List<Integer> addLines, List<Integer> delLines, String actionName, String path, String ClassContent, AbbreviatedObjectId id) {
        TypeDeclaration typeDec = getJavaClass();

        if(!RegrexDefinations.isJava(path)){
            return null;
        }
        if(typeDec==null){
            return SetClassInfo(projectName,null, null, null,
                    addLines, delLines,actionName,path, ClassContent, id);
        }
        List<MethodInfo> methodInfoList = getMethodInfoList();

        return SetClassInfo(projectName,getJavaClass().getName().toString(),getJavaClass().getName().toString(),methodInfoList,
                addLines, delLines, actionName, path, ClassContent, id);
    }

    public MethodInfo getMethodInfo(MethodDeclaration methodDeclaration) {
        MethodInfo methodInfo = new MethodInfo();
        setMethodInfo(methodInfo, methodDeclaration);
        return methodInfo;
    }

    private void setMethodInfo(MethodInfo methodInfo,MethodDeclaration methodDeclaration) {
        methodInfo.setMd5(MD5Encode(methodDeclaration.toString()));
        methodInfo.setMethodName(methodDeclaration.getName().toString());
        methodInfo.setParameters(methodDeclaration.parameters().toString());
    }

    public static String MD5Encode(String s) {
        String MD5String = "";
        try {
            MessageDigest md5 = MessageDigest.getInstance("MD5");
            MD5String = Base64.encodeBase64String(md5.digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return MD5String;
    }


    public static boolean isMethodExist(final MethodDeclaration method, final Map<String, MethodDeclaration> methodsMap) {

        return methodsMap.containsKey(method.getName().toString() + method.parameters().toString());
    }

    public static boolean isMethodTheSame(final MethodDeclaration method1,final MethodDeclaration method2) {
        return MD5Encode(method1.toString()).equals(MD5Encode(method2.toString()));
    }
}