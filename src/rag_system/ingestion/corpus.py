"""NIST-themed teaching paraphrases with local page labels, not verified source excerpts."""

from __future__ import annotations

from rag_system.ingestion.schema import Document, Page


def load_nist_ai_rmf_corpus() -> list[Document]:
    """Load hand-written NIST-themed teaching fixtures, not official PDF pages."""
    doc_core = Document(
        document_id="nist_ai_rmf_1_0",
        title="AI RMF teaching paraphrases (local page labels)",
        source_url="https://doi.org/10.6028/NIST.AI.100-1",
        version="synthetic-fixture-v1",
        pages=[
            Page(
                page_number=1,
                text=(
                    "Executive Summary. The Artificial Intelligence Risk Management Framework "
                    "(AI RMF 1.0) is intended for voluntary use by organizations designing, "
                    "developing, deploying, or using AI systems to help manage the many risks "
                    "of AI and promote trustworthy and responsible AI development and use. "
                    "The Framework is designed to equip organizations and individuals with "
                    "approaches to address risks in AI development and use, helping incorporate "
                    "trustworthiness considerations into AI products, services, and systems."
                ),
            ),
            Page(
                page_number=4,
                text=(
                    "Section 1.1: Audience and Intended Use. AI risks can manifest in diverse "
                    "ways and impact individuals, organizations, and society. The Framework is "
                    "flexible, structured, and non-prescriptive, allowing organizations to "
                    "adapt practices to their specific risk tolerance and operational context. "
                    "AI risk management should be integrated across the entire AI lifecycle, "
                    "from design and pre-training to deployment, monitoring, and decommissioning."
                ),
            ),
            Page(
                page_number=12,
                text=(
                    "Section 2: Characteristics of Trustworthy AI Systems. Trustworthy AI systems "
                    "are characterized by seven interrelated dimensions: Valid and Reliable, "
                    "Safe, Secure and Resilient, Accountable and Transparent, Explainable and "
                    "Interpretable, Privacy-Enhanced, and Fair with Harmful Bias Managed. "
                    "Trade-offs often exist among these characteristics; for example, maximizing "
                    "explainability might require trade-offs with predictive accuracy or privacy."
                ),
            ),
            Page(
                page_number=15,
                text=(
                    "Section 2.1: Valid and Reliable. Validation confirms that a system achieves "
                    "its intended purpose under specified conditions, while reliability is the "
                    "ability of a system to perform consistently over time without unexpected "
                    "failures or systematic degradation across operating environments."
                ),
            ),
            Page(
                page_number=20,
                text=(
                    "Section 3.1: Core Function GOVERN. The GOVERN function cultivates and "
                    "implements a culture of risk management within organizations designing, "
                    "developing, or deploying AI. It establishes governance processes, executive "
                    "oversight, workforce training, risk appetite, and accountability structures."
                ),
            ),
            Page(
                page_number=24,
                text=(
                    "Section 3.2: Core Function MAP. The MAP function establishes the context "
                    "to frame, categorize, and understand risks associated with an AI system. "
                    "It identifies system boundaries, anticipated benefits, potential societal "
                    "harms, target populations, requirements, and impacts on vulnerable groups."
                ),
            ),
            Page(
                page_number=28,
                text=(
                    "Section 3.3: Core Function MEASURE. The MEASURE function employs "
                    "quantitative, qualitative, or hybrid methodologies to assess, analyze, "
                    "and track AI risks and trustworthiness characteristics. It includes testing, "
                    "evaluation, validation, and verification (TEVV) across the system lifecycle."
                ),
            ),
            Page(
                page_number=33,
                text=(
                    "Section 3.4: Core Function MANAGE. The MANAGE function allocates resources to "
                    "identified and measured AI risks on an ongoing basis. It prioritizes risk "
                    "mitigation treatments, implements operational safeguards, defines incident "
                    "response procedures, and monitors residual risks to ensure they remain within "
                    "the organization's acceptable risk tolerance."
                ),
            ),
        ],
    )

    doc_playbook = Document(
        document_id="nist_ai_rmf_playbook",
        title="AI RMF Playbook teaching paraphrases (local page labels)",
        source_url="https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook",
        version="synthetic-fixture-v1",
        pages=[
            Page(
                page_number=5,
                text=(
                    "Playbook Implementation Guidelines: TEVV Workflows. Testing, Evaluation, "
                    "Validation, and Verification (TEVV) tasks should be conducted by independent "
                    "teams whenever feasible to prevent confirmation bias. Pre-deployment TEVV "
                    "should establish quantitative baselines for bias, adversarial robustness, "
                    "and out-of-distribution performance."
                ),
            ),
            Page(
                page_number=18,
                text=(
                    "Playbook Guidance: Managing Generative AI and Foundation Model Risks. "
                    "Generative AI models introduce unique risks, including hallucination, "
                    "data poisoning, privacy leakage, and prompt injection attacks. "
                    "Organizations must maintain provenance tracking, content filtering, "
                    "and human-in-the-loop validation for high-consequence automated decisions."
                ),
            ),
        ],
    )

    return [doc_core, doc_playbook]
