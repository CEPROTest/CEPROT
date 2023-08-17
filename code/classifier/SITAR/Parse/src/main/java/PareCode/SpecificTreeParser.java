package PareCode;

import RoelGeneration.JavaParser;
import RoelGeneration.JavaParserBaseVisitor;
import org.antlr.v4.runtime.misc.Interval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class SpecificTreeParser extends JavaParserBaseVisitor<Void> {
    public static enum ResultEnum {
        PackageDeclaration,ImportDeclaration,NormalClassDeclaration,
        FormalParameterList,Expression,MethodInvocation,ReturnStatement,FieldDeclaration,
        Annotation
    }
    private HashMap<ResultEnum, List<Interval>> hashMap=new HashMap<>();
    @Override
    public Void visitPackageDeclaration(JavaParser.PackageDeclarationContext ctx) {
        var list=new ArrayList<Interval>();

        if(ctx!=null){
            list.add(new Interval(ctx.getStart().getLine(),ctx.getStop().getLine()));
        }

        if(hashMap.get(ResultEnum.PackageDeclaration)==null){
            hashMap.put(ResultEnum.PackageDeclaration,list);
        }else{
            hashMap.get(ResultEnum.PackageDeclaration).addAll(list);
        }
        MergeSolution.merge(hashMap.get(ResultEnum.PackageDeclaration));
        return super.visitPackageDeclaration(ctx);
    }

    @Override
    public Void visitImportDeclaration(JavaParser.ImportDeclarationContext ctx) {
        var list=new ArrayList<Interval>();
        if(ctx!=null){
            list.add(new Interval(ctx.getStart().getLine(),ctx.getStop().getLine()));
        }
        list.add(new Interval(ctx.getStart().getLine(),ctx.getStop().getLine()));
        if(hashMap.get(ResultEnum.ImportDeclaration)==null){
            hashMap.put(ResultEnum.ImportDeclaration,list);
        }else{
            hashMap.get(ResultEnum.ImportDeclaration).addAll(list);
        }
        MergeSolution.merge(hashMap.get(ResultEnum.ImportDeclaration));
        return super.visitImportDeclaration(ctx);
    }

    @Override
    public Void visitClassDeclaration(JavaParser.ClassDeclarationContext ctx) {

        var list=new ArrayList<Interval>();
        list.add(new Interval(ctx.CLASS().getSymbol().getLine(), ctx.CLASS().getSymbol().getLine()));
        list.add(new Interval(ctx.IDENTIFIER().getSymbol().getLine(),ctx.IDENTIFIER().getSymbol().getLine()));
        if(ctx.typeParameters()!=null){
            list.add(new Interval(ctx.typeParameters().getStart().getLine(),ctx.typeParameters().getStop().getLine()));
        }
        if(ctx.EXTENDS()!=null){
            list.add(new Interval(ctx.EXTENDS().getSymbol().getLine(),ctx.EXTENDS().getSymbol().getLine()));
            list.add(new Interval(ctx.typeType().getStart().getLine(),ctx.typeType().getStop().getLine()));
        }
        if(ctx.IMPLEMENTS()!=null){
            list.add(new Interval(ctx.IMPLEMENTS().getSymbol().getLine(), ctx.IMPLEMENTS().getSymbol().getLine()));
            list.add(new Interval(ctx.typeList().getStart().getLine(),ctx.typeList().getStop().getLine()));
        }
        if(hashMap.get(ResultEnum.NormalClassDeclaration)==null){
            hashMap.put(ResultEnum.NormalClassDeclaration,list);
        }else{
            hashMap.get(ResultEnum.NormalClassDeclaration).addAll(list);
        }
        MergeSolution.merge(hashMap.get(ResultEnum.NormalClassDeclaration));
        return super.visitClassDeclaration(ctx);
    }

    @Override
    public Void visitFormalParameterList(JavaParser.FormalParameterListContext ctx) {
        var list=new ArrayList<Interval>();
        if(ctx!=null){
            list.add(new Interval(ctx.getStart().getLine(),ctx.getStop().getLine()));
        }
        if(hashMap.get(ResultEnum.FormalParameterList)==null){
            hashMap.put(ResultEnum.FormalParameterList,list);
        }else{
            hashMap.get(ResultEnum.FormalParameterList).addAll(list);
        }
        MergeSolution.merge(hashMap.get(ResultEnum.FormalParameterList));
        return super.visitFormalParameterList(ctx);
    }


    @Override
    public Void visitMethodCall(JavaParser.MethodCallContext ctx) {
        var list=new ArrayList<Interval>();
        if(ctx!=null){
            list.add(new Interval(ctx.getStart().getLine(),ctx.getStop().getLine()));
        }
        if(hashMap.get(ResultEnum.MethodInvocation)==null){
            hashMap.put(ResultEnum.MethodInvocation,list);
        }else{
            hashMap.get(ResultEnum.MethodInvocation).addAll(list);
        }
        MergeSolution.merge(hashMap.get(ResultEnum.MethodInvocation));
        return super.visitMethodCall(ctx);
    }

    @Override
    public Void visitFieldDeclaration(JavaParser.FieldDeclarationContext ctx) {
        var list=new ArrayList<Interval>();
        if(ctx!=null){
            list.add(new Interval(ctx.getStart().getLine(),ctx.getStop().getLine()));
        }
        if(hashMap.get(ResultEnum.FieldDeclaration)==null){
            hashMap.put(ResultEnum.FieldDeclaration,list);
        }else{
            hashMap.get(ResultEnum.FieldDeclaration).addAll(list);
        }
        MergeSolution.merge(hashMap.get(ResultEnum.FieldDeclaration));
        return super.visitFieldDeclaration(ctx);
    }

    @Override
    public Void visitAnnotation(JavaParser.AnnotationContext ctx) {
        var list=new ArrayList<Interval>();
        if(ctx!=null){
            list.add(new Interval(ctx.getStart().getLine(),ctx.getStop().getLine()));
        }
        if(hashMap.get(ResultEnum.Annotation)==null){
            hashMap.put(ResultEnum.Annotation,list);
        }else{
            hashMap.get(ResultEnum.Annotation).addAll(list);
        }
        MergeSolution.merge(hashMap.get(ResultEnum.Annotation));
        return super.visitAnnotation(ctx);
    }



    public HashMap<ResultEnum, List<Interval>> getHashMap() {
        return hashMap;
    }

    @Override
    public Void visitStatement(JavaParser.StatementContext ctx) {
        var list = new ArrayList<Interval>();
        if(ctx.children==null){
            return super.visitStatement(ctx);
        }
        var Identification = ctx.children.get(0).getText();
        if (Identification.equalsIgnoreCase("return")) {
            list.add(new Interval(ctx.RETURN().getSymbol().getLine(), ctx.RETURN().getSymbol().getLine()));
        }
        if (Identification.equalsIgnoreCase("if") ||
                Identification.equalsIgnoreCase("while")
        ||Identification.equalsIgnoreCase("do")){
            list.add(new Interval(ctx.parExpression().expression().getStart().getLine(), ctx.parExpression().expression().getStop().getLine()));
        }
        if(Identification.equalsIgnoreCase("for")&&ctx.forControl().expression()!=null){
            list.add(new Interval(ctx.forControl().expression().getStart().getLine(),
                    ctx.forControl().expression().getStop().getLine()));
        }
        ResultEnum resultEnum=Identification.equalsIgnoreCase("return")?
                ResultEnum.ReturnStatement: ResultEnum.Expression;


        if (hashMap.get(resultEnum) == null) {
            hashMap.put(resultEnum, list);
        } else {
            hashMap.get(resultEnum).addAll(list);
        }
        MergeSolution.merge(hashMap.get(resultEnum));

        return super.visitStatement(ctx);
    }
}
