package PareCode;

import IntervalTreeInstance.IntegerInterval;
import IntervalTreeInstance.Interval.Bounded;
import IntervalTreeInstance.IntervalTree;
import RoelGeneration.JavaLexer;
import RoelGeneration.JavaParser;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonToken;
import org.antlr.v4.runtime.CommonTokenStream;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.Comment;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.ImportDeclaration;

import java.util.*;

public class PareInstance {
    public static HashMap<SpecificTreeParser.ResultEnum, IntervalTree<Integer>> ParseFile(String content) {
        if(content==null){
            return null;
        }
        CharStream input = CharStreams.fromString(content);
        JavaLexer lexer = new JavaLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);

        JavaParser parser = new JavaParser(tokens);
        JavaParser.CompilationUnitContext compilationUnitContext = parser.compilationUnit();

        SpecificTreeParser tv = new SpecificTreeParser();
        if(compilationUnitContext==null){
            return null;
        }
        tv.visit(compilationUnitContext);

        var hashmap = tv.getHashMap();

        var hashIntervals = new HashMap<SpecificTreeParser.ResultEnum, IntervalTree<Integer>>();

        Iterator iter = hashmap.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry entry = (Map.Entry) iter.next();
            SpecificTreeParser.ResultEnum key = (SpecificTreeParser.ResultEnum) entry.getKey();
            var list = (List<org.antlr.v4.runtime.misc.Interval>) entry.getValue();
            IntervalTree tree = new IntervalTree();
            list.forEach(x -> {
                tree.add(new IntegerInterval(x.a, x.b, Bounded.CLOSED));
            });
            hashIntervals.put(key, tree);
        }
        return hashIntervals;
    }

    public static List<Integer> getCommentLines(String content){
        CharStream input = CharStreams.fromString(content);
        JavaLexer lexer = new JavaLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        List<Integer>  index=new ArrayList<>();
        for(Object o : tokens.getTokens()) {
            CommonToken t = (CommonToken)o;
            if(t.getType() == JavaLexer.COMMENT) {
                System.out.println(t.getCharPositionInLine());
                index.add(t.getLine());
            }
            if(t.getType() == JavaLexer.LINE_COMMENT) {
                System.out.println(t.getCharPositionInLine());
                index.add(t.getLine());
            }

        }
        return index;
    }

    public static CompilationUnit initCompilationUnit(String content) {

        final ASTParser astParser = ASTParser.newParser(8);
        final Map<String, String> options = JavaCore.getOptions();
        JavaCore.setComplianceOptions(JavaCore.VERSION_1_8, options);
        astParser.setCompilerOptions(options);
        astParser.setKind(ASTParser.K_COMPILATION_UNIT);
        astParser.setResolveBindings(true);
        astParser.setBindingsRecovery(true);
        astParser.setStatementsRecovery(true);
        astParser.setSource(content.toCharArray());
        return (CompilationUnit) astParser.createAST(null);
    }

    public static IntervalTree  getIntervalThroughTypes(String content){
        var compilationUnit=initCompilationUnit(content);
        var comments_list=(List<Comment>)compilationUnit.getCommentList();
        var result=new ArrayList<Integer>();
        IntervalTree tree = new IntervalTree();
        comments_list.forEach(x->{
            tree.add(new IntegerInterval(
                    compilationUnit.getLineNumber(x.getStartPosition()),compilationUnit.getLineNumber(x.getStartPosition()+x.getLength()), Bounded.CLOSED));
        });
        return tree;
    }

    public static IntervalTree  getImportsInterval(String content){
        var compilationUnit=initCompilationUnit(content);
        var imports_list=(List<ImportDeclaration>)compilationUnit.imports();
        var result=new ArrayList<Integer>();
        IntervalTree tree = new IntervalTree();
        imports_list.forEach(x->{
            tree.add(new IntegerInterval(
                    compilationUnit.getLineNumber(x.getStartPosition()),compilationUnit.getLineNumber(x.getStartPosition()+x.getLength()), Bounded.CLOSED));
        });
        return tree;
    }

}