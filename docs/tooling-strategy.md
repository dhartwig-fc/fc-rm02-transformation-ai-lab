# Tooling Strategy: 12 AI Repositories → Financial Crime AI Lab

> Review of the "12 GitHub Repos Every AI Developer Should Know" list (Oct 2026), assessed **for this lab specifically**: financial crime, AML, sanctions, regulated data.
> Star counts measure popularity, not fitness. In this domain the ranking criteria are: **data sensitivity, licence, security track record, overlap, and fit to a planned project.**

---

## 1. Executive summary

- **You already run the strongest part of the stack.** Claude Code + Superpowers + Agent Skills is your *build* layer. The next level is building **reusable FinCrime capabilities** that every other tool can plug into, rather than adding more tools.
- **MCP + Skills are the common interface.** Claude Code, Open WebUI, n8n, Dify and LangChain all speak MCP. Build each capability (sanctions lookup, typology knowledge base, case summariser) **once** as an MCP server or Skill, then reuse it across every front-end.
- **Local-first for anything resembling real data.** Ollama is the key unlock. It lets the lab show patterns that a bank's model-risk and data-protection teams could actually approve.
- **Pick one tool per job.** The list contains 3 visual builders (n8n, Dify, Langflow), 2 chat UIs (Open WebUI, Dify), 2 coding agents beyond Claude Code (OpenCode, plus Claude Code itself), and 2 autonomous personal agents (OpenClaw, Hermes). Adopting all of them is tool sprawl.
- **Two repos are "Hold" for a FinCrime context:** OpenClaw and Hermes Agent. They are interesting, but autonomous agents with broad device or chat access and self-modifying skills are hard to govern.

---

## 2. Repository review

Ratings: **Adopt** (use now) · **Trial** (use in a bounded prototype) · **Assess** (read and learn, don't depend on it) · **Hold** (keep out of the lab's core path)

| # | Repo | What it is | Fit for this lab | Licence / risk notes | Verdict |
|---|------|-----------|------------------|----------------------|---------|
| 11 | **anthropics/claude-code** | Agentic coding CLI | Already your core build tool | Commercial product; repo is mainly plugins and issues | **Adopt** (in use) |
| 7 | **anthropics/skills** | Reference Agent Skills + `skill-creator` | Very high. Encode FinCrime method (typologies, SAR narrative structure, screening disposition logic) as Skills | Mixed: most examples are Apache-2.0, document skills are source-available | **Adopt** |
| 6 | **ollama/ollama** | Run open-weight models locally | Very high. Enables PII-safe processing, offline demos, and a cost floor | MIT. Don't expose port 11434 publicly | **Adopt** |
| 5 | **firecrawl/firecrawl** | Website → LLM-ready markdown | High. Adverse media, regulator publications (FCA, OFSI, FATF, FinCEN), and typology report ingestion | **AGPL-3.0** if self-hosted. Respect robots/ToS. Cloud API sends URLs off-box | **Trial** |
| 9 | **open-webui/open-webui** | Self-hosted ChatGPT-style UI | High. Fastest route to a demoable *Knowledge Assistant* over Ollama with built-in RAG and MCP/tool support | Custom licence with a branding-preservation clause (since v0.6.x). Check it before any redistribution | **Trial** |
| 4 | **n8n-io/n8n** | Visual workflow automation + AI nodes | High for *AML Workflow Automation*: alert intake → enrich → triage → route, with human-in-the-loop steps | **Sustainable Use Licence** (fair-code, not OSI). Several critical CVEs in 2025–26: pin versions, patch fast, never internet-expose | **Trial** |
| 12 | **langchain-ai/langchain** (+ LangGraph) | Agent/RAG framework | Medium. Useful for RAG plumbing and LangGraph state machines. Heavy abstraction churn | MIT | **Assess** (choose either this or the Claude Agent SDK as your single code framework) |
| 8 | **langgenius/dify** | All-in-one agentic workflow + RAG platform | Medium. Overlaps with n8n + Open WebUI + LangChain combined | Modified Apache-2.0 with multi-tenant / logo restrictions | **Assess** (choose Dify *or* n8n + Open WebUI, not both) |
| 3 | **anomalyco/opencode** | Open-source, model-agnostic terminal coding agent | Medium. Useful as a second opinion, or to code with **local models** when the code itself is sensitive | MIT | **Assess** |
| 10 | **langflow-ai/langflow** | Drag-and-drop agent builder | Low. Third visual builder. Critical unauthenticated RCE (CVE-2025-3248, on the CISA KEV list) | MIT | **Assess / Hold** |
| 2 | **NousResearch/hermes-agent** | Self-improving agent that writes its own skills | Research interest only. Self-modifying behaviour conflicts with model-risk expectations (SR 11-7 / PRA SS1/23 style: change control, validation, explainability) | MIT | **Hold** (study the ideas in a sandbox) |
| 1 | **openclaw/openclaw** | Personal AI assistant on your device and in your chats | Low for the lab. Broad access to messages, files and shell. Its third-party skill marketplace has had malicious-skill incidents | MIT | **Hold** (never near case or customer data) |

> Licence and CVE notes reflect public information up to mid-2026. **Re-check each repo's LICENSE and security advisories before adopting it.** These details change.

---

## 3. Target architecture: three layers, MCP in the middle

```
┌──────────────────────────────────────────────────────────────────────┐
│ EXPERIENCE LAYER  (how humans use it)                                │
│   Open WebUI (chat / Knowledge Assistant)   n8n (workflows, HITL)    │
└───────────────▲───────────────────────────────────▲──────────────────┘
                │            MCP + Skills            │
┌───────────────┴───────────────────────────────────┴──────────────────┐
│ CAPABILITY LAYER  (the lab's real IP - build once, reuse everywhere) │
│   MCP: sanctions-lookup · typology-kb · entity-resolve · case-store  │
│   Skills: alert-triage · SAR-narrative · adverse-media-assess        │
│   Ingestion: Firecrawl → markdown → vector store                     │
│   Models: Claude (reasoning, synthetic data) · Ollama (sensitive)    │
└───────────────▲──────────────────────────────────────────────────────┘
                │
┌───────────────┴──────────────────────────────────────────────────────┐
│ BUILD LAYER  (how you make it)                                       │
│   Claude Code + Superpowers + frugal routing · anthropics/skills     │
│   (OpenCode + Ollama as the local-only coding fallback)              │
└──────────────────────────────────────────────────────────────────────┘
```

**Design principle:** the capability layer outlives tool choices. If n8n gets replaced by Dify next year, or Open WebUI by something else, the MCP servers and Skills move with you unchanged.

---

## 4. Mapping to the planned projects

| Planned project | Primary tools | First deliverable |
|---|---|---|
| **Financial Crime Knowledge Assistant** | Ollama + Open WebUI (quick win), then a coded RAG pipeline | Chat over public sources: FATF typologies, JMLSG, FCA FCG, OFSI guidance |
| **Sanctions Intelligence Assistant** | Firecrawl + `sanctions-lookup` MCP (on public lists, e.g. OFSI, OFAC SDN, EU, OpenSanctions) + `adverse-media-assess` Skill | Name in → fuzzy matches + adverse media summary with citations |
| **Investigator Copilot** | Claude Code builds it; Skills for triage and narrative; Ollama for any PII-touching step | Synthetic alert → structured triage note + draft SAR narrative (grounds for suspicion, no tipping-off language) |
| **AML Workflow Automation** | n8n calling the MCP servers, with explicit human approval nodes | Synthetic TM alert queue → enrich → risk-score → route to analyst |
| **MCP Financial Crime Toolkit** | `mcp-builder` skill + Claude Code | 3–4 small, tested MCP servers (the capability layer above) |
| **Threat Analysis Explorer** | Firecrawl ingestion + graph analysis (networkx) + LLM typology tagging | Typology tagger over ingested reports, plus a simple network view |

---

## 5. Guardrails (non-negotiable in this domain)

1. **Synthetic or public data only** in this lab repo. Never real customer, alert or SAR data. Generate synthetic alerts and transactions with fixtures.
2. **Sensitivity gate:** add `.claude/frugal-sensitivity.json` with regexes for IBAN, UK sort code + account number, card PAN, passport and NI numbers. This stops sensitive content from being delegated to sub-agents. Route matching work to Ollama.
3. **Human-in-the-loop by design.** Agents *draft* and *recommend*. Humans *decide* on disposition, SAR filing and exits. Model the approval step explicitly in n8n.
4. **Explainability:** every output cites its sources (document, list entry, rule). This is also what makes the work credible to compliance stakeholders.
5. **Supply-chain hygiene:** pin versions, read third-party Skills/MCP servers before installing, never expose n8n, Langflow, Ollama or Open WebUI to the internet, and track CVEs.
6. **Licence register:** record the licence of every adopted repo (AGPL, fair-code and branding clauses matter if the lab output becomes a product).
7. **No self-modifying agents** in anything presented as a control. Skills are versioned in git and reviewed like code.

---

## 6. Roadmap

### Phase 1: Foundations (weeks 1–2)
- [ ] `docker-compose.yml`: Ollama + Open WebUI, localhost only
- [ ] Pull one capable local model and one small embedding model; record laptop RAM/VRAM limits
- [ ] Sensitivity gate config + synthetic data generator (`data/synthetic/`)
- [ ] Licence register (`docs/licences.md`)

### Phase 2: Capability layer (weeks 3–5)
- [ ] First Skill via `skill-creator`: **fincrime-typology-classifier** (input: case facts → FATF/NCA typology + red flags + rationale)
- [ ] First MCP server via `mcp-builder`: **sanctions-lookup** over public list extracts (fuzzy match + scores)
- [ ] Firecrawl ingestion of ~20 public regulator/typology documents → vector store
- [ ] Tests for each (TDD via Superpowers)

### Phase 3: Experience layer (weeks 6–8)
- [ ] Knowledge Assistant in Open WebUI using the ingested corpus + typology Skill
- [ ] n8n workflow: synthetic alert → sanctions-lookup MCP → typology Skill → analyst approval node → case note
- [ ] Short demo video / write-up for portfolio

### Phase 4: Assess and decide (weeks 9–10)
- [ ] Bake-off: the same RAG task in **Dify** vs **n8n + Open WebUI**; keep one
- [ ] Bake-off: LangGraph vs Claude Agent SDK for the Investigator Copilot; keep one
- [ ] Read Hermes Agent's skill-learning loop; write up lessons for "governed self-improvement" (skills proposed by the agent, approved by a human via PR)

---

## 7. What "next level" means

| Superpowers level (now) | Next level |
|---|---|
| Using skills others wrote | **Authoring** domain skills that encode FinCrime expertise |
| One agent in one terminal | Capabilities exposed via **MCP** to many agents and UIs |
| Cloud models for everything | **Model routing by data sensitivity** (Claude vs Ollama) |
| Code experiments | **Governed** prototypes: HITL, citations, licence and CVE discipline |

The differentiator is not knowing 12 repos. It is being the person who can show that agentic AI in financial crime can be **useful and controllable** at the same time.
