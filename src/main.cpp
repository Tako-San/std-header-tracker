#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>
#include <unordered_set>

#include "llvm/Support/CommandLine.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "nlohmann/json.hpp"

bool IsInStdNamespace(const clang::Decl *decl) {
  const auto *ctx = decl->getDeclContext();
  while (ctx) {
    if (const auto *ns = llvm::dyn_cast<clang::NamespaceDecl>(ctx))
      if (ns->isStdNamespace() || ns->getName().contains("__"))
        return true;
    ctx = ctx->getParent();
  }
  return false;
}

bool IsCppStandardHeader(const std::string &path) {
  return (path.find("/c++/") != std::string::npos) ||
         (path.find("/usr/include/") != std::string::npos) ||
         (path.find("include/c++") != std::string::npos);
}

class StdEntityVisitor final
    : public clang::RecursiveASTVisitor<StdEntityVisitor> {
private:
  clang::SourceManager &sm_;

public:
  std::unordered_set<std::string> used_std_entities;

  StdEntityVisitor(clang::SourceManager &SM) : sm_(SM) {}

  bool VisitCallExpr(clang::CallExpr *expr) {
    if (!sm_.isInMainFile(expr->getBeginLoc()))
      return true;

    if (auto *function_decl = expr->getDirectCallee())
      if (IsInStdNamespace(function_decl))
        used_std_entities.insert(function_decl->getQualifiedNameAsString());

    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *expr) {
    if (!sm_.isInMainFile(expr->getBeginLoc()))
      return true;

    if (auto *decl = expr->getDecl())
      if (IsInStdNamespace(decl))
        used_std_entities.insert(decl->getQualifiedNameAsString());

    return true;
  }

  bool VisitTypeLoc(clang::TypeLoc type_loc) {
    if (!sm_.isInMainFile(type_loc.getBeginLoc()))
      return true;

    if (auto *tag_desc = type_loc.getType()->getAsTagDecl())
      if (IsInStdNamespace(tag_desc))
        used_std_entities.insert(tag_desc->getQualifiedNameAsString());

    return true;
  }

  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *expr) {
    if (!sm_.isInMainFile(expr->getBeginLoc()))
      return true;

    if (auto *function_decl = expr->getDirectCallee())
      if (IsInStdNamespace(function_decl))
        used_std_entities.insert(function_decl->getQualifiedNameAsString());

    return true;
  }
};

class UndefinedEntityConsumer final : public clang::DiagnosticConsumer {
public:
  std::unordered_set<std::string> undefined_symbols;

  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic &info) override {
    if (level != clang::DiagnosticsEngine::Error)
      return;

    llvm::SmallString<256> msg_buf;
    info.FormatDiagnostic(msg_buf);
    auto msg = msg_buf.str();

    if (msg.contains("no template named") &&
        msg.contains("in namespace 'std'")) {
      auto start = msg.find("'std::") + 6;
      auto end = msg.find("'", start);
      if (start != llvm::StringRef::npos && end != llvm::StringRef::npos)
        undefined_symbols.insert("std::" + msg.slice(start, end).str());
    } else if (msg.contains("use of undeclared identifier")) {
      auto start = msg.find('\'') + 1;
      auto end = msg.find('\'', start);
      if (start != llvm::StringRef::npos && end != llvm::StringRef::npos)
        undefined_symbols.insert(msg.slice(start, end).str());
    }
  }
};

class MacroTracker final : public clang::PPCallbacks {
private:
  clang::SourceManager &sm_;
  std::unordered_set<std::string> &used_macros_;

public:
  MacroTracker(clang::SourceManager &sm,
               std::unordered_set<std::string> &used_macros)
      : sm_(sm), used_macros_(used_macros) {}

  void MacroExpands(const clang::Token &macro_name_tok,
                    const clang::MacroDefinition &md, clang::SourceRange range,
                    const clang::MacroArgs *) override {
    auto expansion_loc = sm_.getFileLoc(range.getBegin());
    if (sm_.getFileID(expansion_loc) != sm_.getMainFileID())
      return;

    if (!md.getMacroInfo())
      return;

    auto macro_name = macro_name_tok.getIdentifierInfo()->getName().str();

    auto def_loc = md.getMacroInfo()->getDefinitionLoc();
    def_loc = sm_.getFileLoc(def_loc);

    const auto *fe = sm_.getFileEntryForID(sm_.getFileID(def_loc));
    if (fe && IsCppStandardHeader(fe->getName().str()))
      used_macros_.insert(macro_name);

    if (sm_.isInMainFile(range.getBegin()))
      used_macros_.insert(macro_name);
  }
};

class HeaderCollector final : public clang::PPCallbacks {
private:
  clang::SourceManager &sm_;
  std::unordered_set<std::string> &headers_;

public:
  explicit HeaderCollector(clang::SourceManager &sm,
                           std::unordered_set<std::string> &headers)
      : sm_(sm), headers_(headers) {}

  void InclusionDirective(clang::SourceLocation hash_loc, const clang::Token &,
                          clang::StringRef file_name, bool,
                          clang::CharSourceRange, const clang::FileEntry *file,
                          clang::StringRef, clang::StringRef,
                          const clang::Module *,
                          clang::SrcMgr::CharacteristicKind) override {
    if (!file)
      return;

    auto loc = sm_.getFileLoc(hash_loc);
    if (sm_.getFileID(loc) != sm_.getMainFileID())
      return;

    auto full_path = file->getName().str();
    if (IsCppStandardHeader(full_path))
      headers_.insert(file_name.str());
  }
};

class AnalysisConsumer : public clang::ASTConsumer {
  StdEntityVisitor std_visitor_;
  UndefinedEntityConsumer undefined_consumer_;
  clang::CompilerInstance &ci_;
  clang::Preprocessor &pp_;
  std::unordered_set<std::string> macros_;
  std::unordered_set<std::string> &headers_;
  std::unordered_set<std::string> &usages_;

public:
  AnalysisConsumer(clang::CompilerInstance &ci,
                   std::unordered_set<std::string> &headers,
                   std::unordered_set<std::string> &usages)
      : std_visitor_(ci.getSourceManager()), ci_(ci), pp_(ci.getPreprocessor()),
        headers_(headers), usages_(usages) {
    pp_.addPPCallbacks(
        std::make_unique<HeaderCollector>(ci.getSourceManager(), headers_));
    pp_.addPPCallbacks(
        std::make_unique<MacroTracker>(ci.getSourceManager(), macros_));
  }

  void HandleTranslationUnit(clang::ASTContext &ctx) override {
    std_visitor_.TraverseDecl(ctx.getTranslationUnitDecl());

    ci_.getDiagnostics().setClient(&undefined_consumer_,
                                   /*ShouldOwnClient=*/false);

    usages_.merge(std_visitor_.used_std_entities);
    usages_.merge(undefined_consumer_.undefined_symbols);
    usages_.merge(macros_);
  }
};

class AnalysisAction final : public clang::ASTFrontendAction {
public:
  AnalysisAction(std::unordered_set<std::string> &headers,
                 std::unordered_set<std::string> &usages)
      : headers_(headers), usages_(usages) {}

  bool BeginSourceFileAction(clang::CompilerInstance &ci) override {
    ci.getDiagnostics().setErrorLimit(1);
    ci.getDiagnostics().setIgnoreAllWarnings(true);
    ci.getDiagnostics().setSuppressAllDiagnostics(true);
    ci.getDiagnostics().setClient(new clang::IgnoringDiagConsumer());
    return clang::ASTFrontendAction::BeginSourceFileAction(ci);
  }

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<AnalysisConsumer>(CI, headers_, usages_);
  }

private:
  std::unordered_set<std::string> &headers_;
  std::unordered_set<std::string> &usages_;
};

class FrontendActionFactory final
    : public clang::tooling::FrontendActionFactory {
public:
  FrontendActionFactory(std::unordered_set<std::string> &headers,
                        std::unordered_set<std::string> &stdUsages)
      : headers_(headers), usages_(stdUsages) {}

  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<AnalysisAction>(headers_, usages_);
  }

private:
  std::unordered_set<std::string> &headers_;
  std::unordered_set<std::string> &usages_;
};

std::tuple<std::unordered_set<std::string>,
           std::unordered_map<std::string, std::unordered_set<std::string>>,
           std::unordered_set<std::string>>
AnalyzeHeaders(nlohmann::json &json_config,
               std::unordered_set<std::string> &usages,
               std::unordered_set<std::string> &headers) {
  std::unordered_map<std::string, std::unordered_set<std::string>>
      entity_to_headers;
  for (const auto &[header, entities] : json_config.items())
    for (const auto &entity : entities) {
      std::string entityStr = entity.get<std::string>();
      entity_to_headers[entityStr].insert(header);
    }

  std::unordered_set<std::string> redundant_headers;
  for (const auto &header : headers) {
    if (!json_config.contains(header))
      continue;

    const auto &header_entities = json_config[header];
    std::unordered_set<std::string> header_entities_set;
    for (const auto &entity : header_entities)
      header_entities_set.insert(entity.get<std::string>());

    std::unordered_set<std::string> used_usages;
    for (const auto &e : usages)
      if (header_entities_set.count(e))
        used_usages.insert(e);

    for (const auto &e : used_usages)
      usages.erase(e);

    if (used_usages.empty())
      redundant_headers.insert(header);
  }

  std::unordered_map<std::string, std::unordered_set<std::string>>
      missing_headers;
  for (const auto &e : usages) {
    auto it = entity_to_headers.find(e);
    if (it == entity_to_headers.end())
      continue;

    bool any_included = false;
    for (const auto &hdr : it->second)
      if (headers.count(hdr)) {
        any_included = true;
        break;
      }

    if (!any_included)
      missing_headers[e] = it->second;
  }

  std::unordered_set<std::string> unknown_entities;
  for (const auto &entity : usages)
    if (!entity_to_headers.count(entity))
      unknown_entities.insert(entity);

  return {redundant_headers, missing_headers, unknown_entities};
}

void PrintHeaders(
    std::unordered_set<std::string> &redundant_headers,
    std::unordered_map<std::string, std::unordered_set<std::string>>
        &missing_headers,
    std::unordered_set<std::string> &unknown_entities) {
  std::cout << std::endl << "Redundant headers:" << std::endl;
  for (const auto &hdr : redundant_headers)
    std::cout << "  " << hdr << std::endl;

  std::cout << std::endl << "Missing headers:" << std::endl;
  for (const auto &[entity, required_headers] : missing_headers) {
    if (required_headers.empty())
      continue;

    std::cout << "  ";
    for (const auto &hdr : required_headers)
      std::cout << hdr << " ";

    std::cout << std::endl;
  }

  if (!unknown_entities.empty()) {
    std::cout << std::endl
              << "Entities not found in the JSON (possibly non-standard or "
                 "missing entries):"
              << std::endl;
    for (const auto &e : unknown_entities)
      std::cout << "  " << e << std::endl;
  }
}

void CollectSourceFiles(const std::string &path,
                        std::vector<std::string> &files) {
  namespace fs = std::filesystem;
  static const std::unordered_set<std::string> valid_exts = {
      ".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".h", ".c++", ".C"};

  fs::path fs_path(path);

  if (fs::is_regular_file(fs_path)) {
    std::string ext = fs_path.extension().string();
    if (valid_exts.count(ext))
      files.push_back(fs_path.string());
    return;
  }

  if (!fs::is_directory(fs_path))
    return;

  for (const auto &entry : fs::recursive_directory_iterator(fs_path)) {
    if (entry.is_regular_file()) {
      std::string ext = entry.path().extension().string();
      if (valid_exts.count(ext))
        files.push_back(entry.path().string());
    }
  }
}

static llvm::cl::OptionCategory AnalysisCategory("Entity Analysis Options");
static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

static llvm::cl::opt<std::string>
    JSONFilePath("config", llvm::cl::desc("Path to a JSON configuration file"),
                 llvm::cl::value_desc("filepath"), llvm::cl::Required,
                 llvm::cl::cat(AnalysisCategory));

int main(int argc, const char **argv) {
  auto options_parser =
      clang::tooling::CommonOptionsParser::create(argc, argv, AnalysisCategory);

  std::vector<std::string> expanded_paths;
  for (const auto &path : options_parser->getSourcePathList())
    CollectSourceFiles(path, expanded_paths);

  clang::tooling::ClangTool tool(options_parser->getCompilations(),
                                 expanded_paths);
  auto adjuster = getInsertArgumentAdjuster(
      "-w", clang::tooling::ArgumentInsertPosition::BEGIN);
  tool.appendArgumentsAdjuster(adjuster);

  if (expanded_paths.empty()) {
    std::cerr << "No valid source files found" << std::endl;
    return 1;
  }

  std::string json_path = JSONFilePath.getValue();
  if (json_path.empty()) {
    std::cerr << "JSON file path is empty" << std::endl;
    return 1;
  }

  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    std::cerr << "Failed to open JSON file: " << json_path << std::endl;
    return 1;
  }

  nlohmann::json json_config;
  json_file >> json_config;

  std::unordered_set<std::string> usages;
  std::unordered_set<std::string> headers;
  int result = tool.run(new FrontendActionFactory(headers, usages));

  auto [redundant_headers, missing_headers, unknown_entities] =
      AnalyzeHeaders(json_config, usages, headers);

  PrintHeaders(redundant_headers, missing_headers, unknown_entities);

  return result;
}