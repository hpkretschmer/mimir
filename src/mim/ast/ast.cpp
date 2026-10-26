#include "mim/ast/ast.h"

#include <set>

#include "mim/ast/parser.h"

using namespace std::literals;

namespace mim::ast {

// Node map: Parser::import holds the slot across the nested parses that may insert further entries.
struct AST::Files : std::unordered_map<const fe::Src*, Ptr<File>> {};

AST::AST(World& world)
    : world_(&world)
    , files_(std::make_unique<Files>()) {}

AST::AST(AST&& other)
    : AST(other.world()) {
    swap(*this, other);
}

AST::~AST() = default;

std::pair<Ptr<File>&, bool> AST::file(const fe::Src* src) {
    auto [i, fresh] = files_->try_emplace(src);
    return {i->second, fresh};
}

namespace {

constexpr auto Cpp_Keywords = std::to_array<std::string_view>({
    "alignas",     "alignof",   "and",       "and_eq",    "asm",      "auto",         "bitand",
    "bitor",       "bool",      "break",     "case",      "catch",    "char",         "char8_t",
    "char16_t",    "char32_t",  "class",     "compl",     "concept",  "const",        "const_cast",
    "consteval",   "constexpr", "constinit", "continue",  "co_await", "co_return",    "co_yield",
    "decltype",    "default",   "delete",    "do",        "double",   "dynamic_cast", "else",
    "enum",        "explicit",  "export",    "extern",    "false",    "float",        "for",
    "friend",      "goto",      "if",        "inline",    "int",      "long",         "mutable",
    "namespace",   "new",       "noexcept",  "not",       "not_eq",   "nullptr",      "operator",
    "or",          "or_eq",     "private",   "protected", "public",   "register",     "reinterpret_cast",
    "requires",    "return",    "short",     "signed",    "sizeof",   "static",       "static_assert",
    "static_cast", "struct",    "switch",    "template",  "this",     "thread_local", "throw",
    "true",        "try",       "typedef",   "typeid",    "typename", "union",        "unsigned",
    "using",       "virtual",   "void",      "volatile",  "wchar_t",  "while",        "xor",
    "xor_eq",
});

constexpr auto Py_Keywords = std::to_array<std::string_view>({
    "False", "None",     "True",  "and",    "as",   "assert", "async",  "await",    "break",
    "class", "continue", "def",   "del",    "elif", "else",   "except", "finally",  "for",
    "from",  "global",   "if",    "import", "in",   "is",     "lambda", "nonlocal", "not",
    "or",    "pass",     "raise", "return", "try",  "while",  "with",   "yield",
});

/// @p sym as an identifier of the target language: a keyword gets a trailing `_`.
std::string escape(Sym sym, std::span<const std::string_view> keywords) {
    auto id = std::string(sym.view());
    return std::ranges::contains(keywords, sym.view()) ? id + "_" : id;
}

std::string_view kind(const AnnexInfo& annex) { return annex.fields ? "field" : "ctor"; }

/// Translates the `norm` rules sharing a name into the C++ normalizer of that name.
class NormGen {
public:
    /// @p plugins receives every plugin whose annexes the generated code refers to.
    NormGen(AST& ast, std::set<std::string>& plugins)
        : ast_(ast)
        , plugins_(plugins) {}

    /// Each rule becomes its own function, so a failed match can just `return nullptr` and the next rule is tried.
    void emit(std::ostream& os, Sym name, const std::vector<const RuleDecl*>& rules) {
        auto norm = escape(name, Cpp_Keywords);
        for (size_t i = 0, e = rules.size(); i != e; ++i)
            emit(os, std::format("{}_{}", norm, i), rules[i]);

        std::println(os, "inline const Def* {}(const Def* type, const Def* callee, const Def* arg) {{", norm);
        for (size_t i = 0, e = rules.size(); i != e; ++i)
            std::println(os, "    if (auto res = {}_{}(type, callee, arg)) return res;", norm, i);
        std::println(os, "    return nullptr;\n}}\n");
    }

private:
    void emit(std::ostream& os, const std::string& fn, const RuleDecl* rule) {
        metas_.clear();
        bound_.clear();
        body_.str("");
        num_tmps_ = 0;
        collect(rule->var());

        // TODO a non-App lhs, such as a nullary axiom, has no (callee, arg) to match against
        if (auto app = rule->lhs()->isa<AppExpr>()) {
            match(app->callee(), "callee");
            match(app->arg(), "arg");
        } else {
            ast_.error().e(rule->lhs()->loc(), "the left-hand side of a `norm` rule must be an application");
        }

        if (auto prim = rule->guard()->isa<PrimaryExpr>(); !prim || prim->tag() != Tok::Tag::K_tt)
            line("if ({} != w.lit_tt()) return nullptr;", build(rule->guard()));
        // TODO guard against an rhs that rebuilds the lhs, which recurses into this very normalizer forever
        line("return {};", build(rule->rhs()));

        std::println(os,
                     "inline const Def* {}(const Def* type, [[maybe_unused]] const Def* callee, "
                     "[[maybe_unused]] const Def* arg) {{",
                     fn);
        std::println(os, "    [[maybe_unused]] auto& w = type->world();");
        std::print(os, "{}", body_.str());
        std::println(os, "}}\n");
    }

    void collect(const Ptrn* ptrn) {
        if (auto tuple = ptrn->isa<TuplePtrn>()) {
            for (const auto& p : tuple->ptrns())
                collect(p.get());
        } else if (ptrn->isa<AliasPtrn>()) {
            // TODO bind the alias to the whole matched tuple
            ast_.error().e(ptrn->loc(), "`as` patterns are not yet supported in a `norm` rule");
        } else if (auto dbg = ptrn->dbg(); !dbg.is_anon()) {
            metas_[ptrn] = std::string(dbg.sym().view()) + "_";
        }
    }

    void match(const Expr* expr, const std::string& def) {
        if (auto path = expr->isa<PathExpr>()) {
            if (auto i = metas_.find(path->decl()); i != metas_.end()) {
                // TODO check that def inhabits the meta variable's type
                if (bound_.emplace(path->decl()).second)
                    line("auto {} = {};", i->second, def);
                else
                    line("if ({} != {}) return nullptr;", def, i->second);
                return;
            }
        } else if (expr->isa<HoleExpr>()) {
            return;
        } else if (auto app = expr->isa<AppExpr>()) {
            // TODO an implicit argument is an App in the IR but invisible in the AST, so both shapes diverge
            auto tmp = std::format("app{}", num_tmps_++);
            line("auto {} = {}->isa<App>();", tmp, def);
            line("if (!{}) return nullptr;", tmp);
            match(app->callee(), tmp + "->callee()");
            match(app->arg(), tmp + "->arg()");
            return;
        } else if (auto tuple = expr->isa<TupleExpr>()) {
            auto tmp = std::format("tup{}", num_tmps_++);
            auto n   = tuple->num_elems();
            line("auto {} = {};", tmp, def);
            line("if (Lit::isa({}->arity()) != {}_n) return nullptr;", tmp, n);
            for (size_t i = 0; i != n; ++i)
                match(tuple->elem(i), std::format("{}->proj({}, {})", tmp, n, i));
            return;
        }

        // TODO match up to normalization: e.g. a literal arg may stem from a folded application in the pattern
        line("if ({} != {}) return nullptr;", def, build(expr));
    }

    std::string build(const Expr* expr) {
        if (auto path = expr->isa<PathExpr>()) {
            auto decl = path->decl();
            if (auto i = metas_.find(decl); i != metas_.end()) {
                if (!bound_.contains(decl))
                    ast_.error().e(expr->loc(), "meta variable `{}` is not bound by the left-hand side", path->dbg());
                return i->second;
            }
            if (auto [annex, sub] = decl->annex_sub(); annex) return build(*annex, sub);
            // TODO other declarations of the plugin, e.g. a `let` or a `lam`
        } else if (auto lit = expr->isa<LitExpr>(); lit && !lit->type()) {
            // TODO typed, signed, float, char, and string literals as well as ⊥ and ⊤
            if (lit->tag() == Tok::Tag::L_u) return std::format("w.lit_nat({}_n)", lit->tok().lit_u());
            if (lit->tag() == Tok::Tag::L_i) {
                auto [size, val] = lit->tok().lit_i();
                return std::format("w.lit_idx({}_n, {}_n)", size, val);
            }
        } else if (auto prim = expr->isa<PrimaryExpr>()) {
            // TODO the remaining primaries
            switch (prim->tag()) {
                case Tok::Tag::K_tt: return "w.lit_tt()";
                case Tok::Tag::K_ff: return "w.lit_ff()";
                case Tok::Tag::K_Nat: return "w.type_nat()";
                case Tok::Tag::K_Bool: return "w.type_bool()";
                default: break;
            }
        } else if (auto app = expr->isa<AppExpr>()) {
            return std::format("w.implicit_app({}, {})", build(app->callee()), build(app->arg()));
        } else if (auto tuple = expr->isa<TupleExpr>()) {
            auto elems = std::string();
            for (const auto& elem : tuple->elems())
                elems += (elems.empty() ? "" : ", ") + build(elem.get());
            return std::format("w.tuple({{{}}})", elems);
        }

        // TODO infix operators, extracts, inserts, `let`, lambdas, sigmas, ...
        ast_.error().e(expr->loc(), "expression not yet supported in a `norm` rule");
        return "nullptr";
    }

    std::string build(const AnnexInfo& annex, sub_t sub) {
        plugins_.emplace(annex.sym.plugin.str());
        auto tag = std::format("plug::{}::{}", annex.sym.plugin, escape(annex.sym.tag, Cpp_Keywords));
        if (annex.subs.empty()) return std::format("w.annex<{}>()", tag);
        return std::format("w.annex({}::{})", tag, escape(annex.subs[sub].front(), Cpp_Keywords));
    }

    template<class... Args>
    void line(std::format_string<Args...> fmt, Args&&... args) {
        std::println(body_, "    {}", std::format(fmt, std::forward<Args>(args)...));
    }

    AST& ast_;
    std::set<std::string>& plugins_;
    ankerl::unordered_dense::map<const Decl*, std::string> metas_;
    ankerl::unordered_dense::set<const Decl*> bound_;
    std::ostringstream body_;
    size_t num_tmps_ = 0;
};

} // namespace

void AST::bootstrap(Sym plugin, std::ostream& h) {
    auto targets = std::set<std::string>();

    // Generate the rule-based normalizers first, so an unsupported rule fails before anything is written.
    auto rule_norms = std::vector<std::pair<Sym, std::vector<const RuleDecl*>>>();
    for (auto rule : plugin2norms(plugin)) {
        auto name = rule->dbg().sym();
        auto i    = std::ranges::find(rule_norms, name, [](const auto& p) { return p.first; });
        if (i == rule_norms.end()) i = rule_norms.insert(i, {name, {}});
        i->second.emplace_back(rule);
    }
    auto norms_os = std::ostringstream();
    auto norm_gen = NormGen(*this, targets);
    for (const auto& [name, rules] : rule_norms)
        norm_gen.emit(norms_os, name, rules);
    auto is_rule_norm = [&](Dbg norm) {
        return std::ranges::contains(rule_norms, norm.sym(), [](const auto& p) { return p.first; });
    };

    // TODO a family needs a `template<tag>` normalizer, generated from the rules whose lhs head is that sub-tag
    for (const auto& [_, annex] : plugin2annexes(plugin))
        if (annex.sym.plugin == plugin && !annex.subs.empty() && annex.normalizer && is_rule_norm(annex.normalizer))
            error().e(annex.normalizer.loc(), "`norm` rules for an axiom family are not yet supported");
    error().ack();
    targets.erase(plugin.str());

    auto tab = fe::Tab::spaces();
    std::println(h, "{}#pragma once\n", tab);
    std::println(h, "{}#include <mim/axm.h>", tab);
    std::println(h, "{}#include <mim/plugin.h>", tab);
    if (!rule_norms.empty()) std::println(h, "{}#include <mim/world.h>", tab);
    std::println(h, "");

    const auto& foreign = plugin2foreign_aliases(plugin);
    for (const auto& alias : foreign)
        targets.emplace(alias.annex->sym.plugin.str());
    for (const auto& target : targets)
        std::println(h, "{}#include <mim/plug/{}/autogen.h>", tab, target);
    if (!targets.empty()) std::println(h, "");

    std::println(h, "{}/// @namespace mim::plug::{} @ref {}", tab, plugin, plugin);
    std::println(h, "{}namespace mim {{", tab);
    std::println(h, "{}namespace plug::{} {{\n", tab, plugin);

    plugin_t plugin_id = *Annex::mangle(plugin);
    std::vector<std::ostringstream> normalizers, outer_namespace;

    std::println(h, "{}static constexpr plugin_t Plugin_Id = 0x{:x};\n", tab, plugin_id);

    for (const auto& [alias, annex, sub] : foreign) {
        auto id  = escape(alias, Cpp_Keywords);
        auto tag = escape(annex->sym.tag, Cpp_Keywords);
        if (annex->subs.empty())
            std::println(h, "{}using {} = plug::{}::{};", tab, id, annex->sym.plugin, tag);
        else
            std::println(h, "{}static constexpr auto {} = plug::{}::{}::{};", tab, id, annex->sym.plugin, tag,
                         escape(annex->subs[sub].front(), Cpp_Keywords));
    }
    if (!foreign.empty()) std::println(h, "");

    const auto& unordered = plugin2annexes(plugin);
    std::deque<std::pair<Sym, AnnexInfo>> infos(unordered.begin(), unordered.end());
    std::ranges::sort(infos, [&](const auto& p1, const auto& p2) { return p1.second.id.tag < p2.second.id.tag; });

    // clang-format off
    for (const auto& [key, annex] : infos) {
        const auto& sym = annex.sym;
        if (sym.plugin != plugin) continue; // this is from an import
        auto tag = escape(sym.tag, Cpp_Keywords);

        std::println(h, "{}/// @name `{}.{}`\n///@{{", tab, plugin, sym.tag);
        std::println(h, "{}enum class {} : flags_t {{", tab, tag);
        ++tab;
        flags_t ax_id = annex.base();

        auto& os = outer_namespace.emplace_back();
        std::print(os, "template<> constexpr flags_t Annex::Base<plug::{}::{}> = 0x{:x};\n", plugin, tag, ax_id);

        if (auto& subs = annex.subs; !subs.empty()) {
            for (const auto& aliases : subs) {
                auto id = ax_id++;
                for (const auto alias : aliases)
                    std::println(h, "{}{} = 0x{:x},", tab, escape(alias, Cpp_Keywords), id);

                if (auto norm = annex.normalizer) {
                    auto sub = escape(aliases.front(), Cpp_Keywords);
                    auto& os = normalizers.emplace_back();
                    std::print(os, "normalizers[flags_t({}::{})] = &{}<{}::{}>;", tag, sub, norm, tag, sub);
                }
            }
        } else {
            if (auto norm = annex.normalizer)
                std::print(normalizers.emplace_back(), "normalizers[flags_t(Annex::Base<{}>)] = &{};", tag, norm);
        }
        --tab;
        std::println(h, "{}}};\n", tab);

        std::println(outer_namespace.emplace_back(), "template<> constexpr size_t Annex::Num<plug::{}::{}> = {};", plugin, tag, annex.subs.size());

        if (auto norm = annex.normalizer; norm && !is_rule_norm(norm)) {
            if (auto& subs = annex.subs; !subs.empty()) {
                std::println(h, "{}template<{}>\nconst Def* {}(const Def*, const Def*, const Def*);\n", tab, tag,
                           norm);
            } else {
                std::println(h, "{}const Def* {}(const Def*, const Def*, const Def*);", tab, norm);
            }
        }

        if (!annex.names.empty()) {
            std::println(h, "{}/// The {}s of `{}.{}`, by index.", tab, kind(annex), plugin, sym.tag);
            std::print(h, "{}struct {}_{} {{ enum : nat_t {{", tab, sym.tag, kind(annex));
            for (size_t i = 0, e = annex.names.size(); i != e; ++i)
                if (auto name = annex.names[i]) std::print(h, " {} = {},", escape(name, Cpp_Keywords), i);
            std::println(h, " }}; }};\n");
        }
        std::println(h, "{}///@}}\n", tab);
    }
    // clang-format on

    if (!normalizers.empty()) {
        std::println(h, "{}void register_normalizers(Normalizers& normalizers);\n", tab);
        std::println(h, "{}#define MIM_{}_NORMALIZER_IMPL \\", tab, plugin);
        ++tab;
        std::println(h, "{}void register_normalizers(Normalizers& normalizers) {{\\", tab);
        ++tab;
        for (const auto& normalizer : normalizers)
            std::println(h, "{}{} \\", tab, normalizer.str());
        --tab;
        std::println(h, "{}}}", tab);
        --tab;
    }

    std::println(h, "{}}} // namespace plug::{}\n", tab, plugin);

    std::println(h, "{}#ifndef DOXYGEN // don't include in Doxygen documentation\n", tab);
    for (const auto& line : outer_namespace)
        std::print(h, "{}{}", tab, line.str());
    std::println(h, "{}", tab);

    // emit helpers for non-function axm
    for (const auto& [tag, ax] : infos) {
        auto sym = ax.sym;
        if ((ax.pi && *ax.pi) || sym.plugin != plugin) continue; // from function or other plugin?
        std::println(h, "{}template<> struct Axm::IsANode<plug::{}::{}> {{ using type = Axm; }};", tab, sym.plugin,
                     escape(sym.tag, Cpp_Keywords));
    }

    std::println(h, "{}\n#endif", tab);
    std::println(h, "{}}} // namespace mim\n", tab);

    std::println(h, "{}#ifndef DOXYGEN // don't include in Doxygen documentation\n", tab);
    for (const auto& [key, annex] : infos) {
        if (!annex.subs.empty()) {
            auto sym = annex.sym;
            std::println(h, "{}template<> struct fe::is_bit_enum<mim::plug::{}::{}> : std::true_type {{}};", tab,
                         sym.plugin, escape(sym.tag, Cpp_Keywords));
        }
    }

    std::println(h, "{}\n#endif", tab);

    // Only here are all Annex specializations, which the generated normalizers instantiate, complete.
    if (!rule_norms.empty()) {
        std::println(h, "\nnamespace mim::plug::{} {{\n", plugin);
        std::print(h, "{}", norms_os.str());
        std::println(h, "}} // namespace mim::plug::{}", plugin);
    }
}

void AST::bootstrap_py(Sym plugin, std::ostream& h) {
    fe::Tab tab;
    plugin_t plugin_id = *Annex::mangle(plugin);

    std::print(h, "from enum import IntEnum\n\n");
    std::println(h, "class {}(IntEnum):", plugin);
    ++tab;
    std::println(h, "{}ID = 0x{:x}", tab, plugin_id);
    std::vector<const AnnexInfo*> annexes_with_subs;

    const auto& unordered = plugin2annexes(plugin);
    std::deque<std::pair<Sym, AnnexInfo>> infos(unordered.begin(), unordered.end());
    std::ranges::sort(infos, [&](const auto& p1, const auto& p2) { return p1.second.id.tag < p2.second.id.tag; });
    for (const auto& [key, annex] : infos) {
        const auto& sym = annex.sym;
        if (sym.plugin != plugin) continue;

        flags_t ax_id = annex.base();

        if (auto& subs = annex.subs; subs.empty())
            std::println(h, "{}{} = 0x{:x}", tab, escape(sym.tag, Py_Keywords), ax_id);
        else
            annexes_with_subs.emplace_back(&annex);
    }
    for (const auto& [alias, annex, sub] : plugin2foreign_aliases(plugin))
        std::println(h, "{}{} = 0x{:x}", tab, escape(alias, Py_Keywords), annex->base() + sub);
    std::print(h, "\n");
    --tab;

    for (auto annex : annexes_with_subs) {
        flags_t ax_id = annex->base();
        auto tag      = escape(annex->sym.tag, Py_Keywords);
        std::println(h, "class _{}_{}(IntEnum):", plugin, tag);
        ++tab;

        for (const auto& aliases : annex->subs) {
            auto id = ax_id++;
            for (const auto alias : aliases)
                std::println(h, "{}{} = 0x{:x}", tab, escape(alias, Py_Keywords), id);
        }

        --tab;
        std::println(h, "\n{}.{} = _{}_{}\n", plugin, tag, plugin, tag);
    }

    for (const auto& [key, annex] : infos) {
        if (annex.sym.plugin != plugin || annex.names.empty()) continue;
        auto type = std::format("{}_{}", annex.sym.tag, kind(annex));
        std::println(h, "class _{}_{}(IntEnum):", plugin, type);
        ++tab;
        for (size_t i = 0, e = annex.names.size(); i != e; ++i)
            if (auto name = annex.names[i]) std::println(h, "{}{} = {}", tab, escape(name, Py_Keywords), i);
        --tab;
        std::println(h, "\n{}.{} = _{}_{}\n", plugin, type, plugin, type);
    }
}

/*
 * Other
 */

Sym TuplePtrn::field(size_t i) const {
    auto p = ptrn(i);
    return (p->isa<IdPtrn>() || p->isa<GrpPtrn>()) && !p->dbg().is_anon() ? p->dbg().sym() : Sym();
}

LamExpr::LamExpr(Ptr<LamDecl> lam)
    : Expr(lam->loc())
    , lam_(lam) {}

/*
 * Ptrn::to_expr
 */

Ptr<Expr> Ptrn::to_expr(AST& ast, Ptr<Ptrn> ptrn) {
    if (auto idp = ptrn->isa<IdPtrn>(); idp && !idp->dbg() && idp->type()) {
        if (auto pe = idp->type()->isa<PathExpr>())
            return ast.ptr<PathExpr>(ast.ptr<Path>(pe->path()->loc(), pe->path()->dbgs()));
    } else if (auto tuple = ptrn->isa<TuplePtrn>(); tuple && tuple->is_brckt()) {
        return ast.ptr<SigmaExpr>(Ptr<TuplePtrn>(tuple));
    }
    return {};
}

void File::compile(AST& ast) const {
    bind(ast);
    ast.error().ack();
    emit(ast);
    ast.error().report();
}

AST load_plugins(World& world, fe::View<std::string> plugins) {
    auto tag    = world.driver().flags().bootstrap ? Tok::Tag::K_import : Tok::Tag::K_plugin;
    auto ast    = AST(world);
    auto parser = Parser(ast);

    if (!plugins.empty()) {
        auto imports = parser.import_plugins(plugins, tag);
        auto decls   = Ptrs<ValDecl>();
        for (auto import : imports)
            decls.emplace_back(import);

        // No Loc: this File spans no source, and hulling the imports would mix Loc%s of different files.
        auto file = ast.ptr<File>(Loc(), ast.scope(), ast.copy(decls));
        file->compile(ast);
    }

    return ast;
}

} // namespace mim::ast
