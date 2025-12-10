# Use Cases

This document provides comprehensive guidance on implementing TinyForgeAI across various business scenarios, from quick wins to enterprise-grade deployments.

## How to Pick the Right Use Case

Before diving in, use this decision checklist:

1. **Data Availability**: Do you have labeled Q&A pairs, documents, or conversation logs? Start with use cases matching your data format.
2. **Latency Requirements**: Need sub-100ms responses? Consider edge deployment or quantized models. Batch processing? Standard deployment works fine.
3. **Privacy Requirements**: Handling PII, PHI, or classified data? Prioritize on-prem/air-gapped deployment with audit logging enabled.

---

## 1. Internal Knowledge Assistant

Build an AI assistant that answers employee questions using internal documentation, policies, and procedures.

### Basic

**Scenario**: A small team wants to reduce repetitive Slack questions about company policies by deploying a simple Q&A bot trained on their handbook.

**Example Dataset** (`examples/use_cases/knowledge_assistant_sample.jsonl`):
```json
{"input": "What is the PTO policy?", "output": "Employees receive 20 days of PTO per year, accrued monthly. Unused days carry over up to 5 days.", "metadata": {"source": "handbook", "section": "benefits"}}
{"input": "How do I submit expenses?", "output": "Submit expenses via Concur within 30 days. Attach receipts for amounts over $25. Manager approval required.", "metadata": {"source": "handbook", "section": "finance"}}
```

**CLI Example**:
```bash
# Train a small knowledge model
foremforge train \
  --data examples/use_cases/knowledge_assistant_sample.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output models/knowledge-assistant \
  --epochs 3 \
  --dry-run  # Remove for actual training
```

**API Example**:
```python
from fastapi.testclient import TestClient
from inference_server.main import app

client = TestClient(app)
response = client.post("/predict", json={
    "input": "What is the PTO policy?",
    "parameters": {"max_tokens": 100}
})
print(response.json()["output"])
```

**Deployment Notes**: Run on a single VM with 8GB RAM. Expect 200-500ms latency. Use INT8 quantization for faster inference.

**Success Metrics**:
- Answer accuracy: >85% on test set
- Query resolution rate: >70% without escalation
- Average response time: <500ms

**Security & Compliance**:
- Enable audit logging for all queries
- Restrict access via API keys per department
- No PII in training data; redact names/emails before training

### Advanced

**Scenario**: Enterprise deployment serving 10,000+ employees with role-based access, RAG integration, and continuous learning from feedback.

**CLI Example**:
```bash
# Train with LoRA for efficient updates
foremforge train \
  --data ./data/knowledge_base_full.jsonl \
  --base-model microsoft/phi-2 \
  --use-lora --lora-r 16 --lora-alpha 32 \
  --output models/knowledge-assistant-v2 \
  --epochs 5
```

**API Example with RAG**:
```python
from backend.rag import RAGPipeline

pipeline = RAGPipeline(
    embedder_model="sentence-transformers/all-MiniLM-L6-v2",
    generator_model="models/knowledge-assistant-v2",
    vector_store_path="./data/knowledge_vectors"
)
response = pipeline.query("How do I request FMLA leave?", top_k=3)
print(f"Answer: {response['answer']}\nSources: {response['sources']}")
```

**Deployment Notes**: Deploy on Kubernetes with 3 replicas. Use Redis for caching frequent queries. Latency target: <200ms p95.

**Success Metrics**:
- Precision@3 for retrieval: >0.8
- User satisfaction (thumbs up): >90%
- Cost per 1,000 queries: <$0.10

**Security & Compliance**:
- Implement SSO integration for user authentication
- Role-based document filtering (HR docs visible only to HR)
- Encrypt embeddings at rest; rotate encryption keys quarterly
- GDPR compliance: implement right-to-deletion for user queries

**Advanced Tips**:
- Use A/B testing to compare LoRA adapters before full rollout
- Implement incremental LoRA updates weekly from user feedback
- Set up model rollback triggers on accuracy degradation >5%
- Hybrid RAG: combine vector search with BM25 for better recall

---

## 2. Customer Support Automation

Automate tier-1 support responses and route complex issues to human agents.

### Basic

**Scenario**: A SaaS startup wants to auto-respond to common support tickets (password resets, billing questions) while routing complex issues.

**Example Dataset** (`examples/use_cases/support_automation_sample.jsonl`):
```json
{"input": "I forgot my password and cannot log in", "output": "I can help you reset your password. Click the 'Forgot Password' link on the login page, enter your email, and follow the instructions sent to your inbox. The reset link expires in 24 hours.", "metadata": {"category": "account", "priority": "low"}}
{"input": "I was charged twice for my subscription this month", "output": "I apologize for the billing error. I have initiated a refund for the duplicate charge. It will appear in your account within 3-5 business days. Your subscription remains active.", "metadata": {"category": "billing", "priority": "high"}}
```

**CLI Example**:
```bash
foremforge train \
  --data examples/use_cases/support_automation_sample.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output models/support-bot \
  --dry-run
```

**API Example**:
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "input": "How do I cancel my subscription?",
    "parameters": {"max_tokens": 150, "temperature": 0.3}
})
print(response.json()["output"])
```

**Deployment Notes**: Deploy behind your helpdesk (Zendesk, Intercom). Use webhook integration. Target <2s response time for ticket auto-replies.

**Success Metrics**:
- Ticket deflection rate: >40%
- First response time: <30 seconds
- Customer satisfaction (CSAT): >4.0/5.0

**Security & Compliance**:
- Mask credit card numbers and SSNs in inputs
- Log all auto-responses for audit trails
- Implement human handoff trigger for sensitive topics

### Advanced

**Scenario**: Multi-channel support (email, chat, phone transcripts) with sentiment detection, priority routing, and agent assist features.

**CLI Example**:
```bash
foremforge train \
  --data ./data/support_full_dataset.jsonl \
  --base-model microsoft/phi-2 \
  --use-lora --lora-r 32 \
  --output models/support-agent-assist \
  --epochs 5 \
  --gradient-checkpointing
```

**Deployment Notes**: Deploy on GPU instances for real-time agent assist. Use streaming responses for chat. Implement fallback to human queue on low confidence (<0.7).

**Success Metrics**:
- Resolution time reduction: >30%
- Agent productivity increase: >25%
- Escalation accuracy: >95%

**Security & Compliance**:
- PCI-DSS compliance for payment-related queries
- Implement data retention policies (90-day auto-delete)
- Geographic data residency for EU customers

**Advanced Tips**:
- Train separate LoRA adapters per product line
- Implement confidence-based routing (high confidence: auto-respond, low: human review)
- A/B test response templates for CSAT optimization

---

## 3. Domain-Specific Summarization

Generate concise summaries of long documents, reports, or articles tailored to your domain.

### Basic

**Scenario**: A research team wants to summarize academic papers into 2-3 sentence abstracts for quick scanning.

**Example Dataset** (`examples/use_cases/summarization_sample.txt`):
```
Document: The quarterly financial report indicates a 15% increase in revenue compared to the previous quarter, driven primarily by strong performance in the enterprise segment. Operating expenses remained flat, resulting in improved profit margins. The company projects continued growth in the next quarter based on the current sales pipeline and recent product launches.

Summary: Q3 revenue grew 15% quarter-over-quarter, led by enterprise sales. Flat operating costs improved margins. Growth expected to continue based on pipeline strength.
```

**CLI Example**:
```bash
foremforge train \
  --data ./data/summarization_pairs.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output models/summarizer \
  --max-length 512 \
  --dry-run
```

**API Example**:
```python
response = client.post("/predict", json={
    "input": f"Summarize: {long_document}",
    "parameters": {"max_tokens": 100, "temperature": 0.2}
})
summary = response.json()["output"]
```

**Deployment Notes**: Batch processing suitable. Run overnight for large document sets. Memory: 4GB minimum.

**Success Metrics**:
- ROUGE-L score: >0.4
- Human preference rate: >70% vs. generic summarizers
- Processing time: <5s per document

**Security & Compliance**:
- Ensure no confidential content leaks in summaries
- Implement document classification before summarization
- Audit log all summarized documents

### Advanced

**Scenario**: Real-time meeting summarization with action item extraction and participant attribution.

**Deployment Notes**: Use streaming inference for live transcription. Deploy on edge for low-latency meeting rooms. GPU recommended for real-time.

**Success Metrics**:
- Action item extraction accuracy: >90%
- Summary completeness rating: >4.5/5
- Latency: <3s after meeting end

**Advanced Tips**:
- Train separate models for different meeting types (standup, planning, review)
- Implement speaker diarization preprocessing
- Use incremental summarization for long meetings (summarize every 10 minutes)

---

## 4. Internal Workflow & SOP Automation

Automate standard operating procedures and workflow guidance.

### Basic

**Scenario**: Help desk staff need quick answers about IT procedures (password resets, VPN setup, software installation).

**Example Dataset** (`examples/use_cases/sop_workflow_sample.jsonl`):
```json
{"input": "How do I set up VPN on my Mac?", "output": "1. Download GlobalProtect from the IT portal. 2. Install and open the application. 3. Enter vpn.company.com as the portal address. 4. Sign in with your SSO credentials. 5. Click Connect.", "metadata": {"category": "IT", "sop_id": "IT-VPN-001"}}
{"input": "What is the onboarding checklist for new hires?", "output": "Day 1: Complete I-9 form, set up laptop, activate email. Week 1: Complete security training, meet team, review handbook. Month 1: Complete compliance training, first 1:1 with manager.", "metadata": {"category": "HR", "sop_id": "HR-ONB-001"}}
```

**CLI Example**:
```bash
foremforge train \
  --data examples/use_cases/sop_workflow_sample.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output models/sop-assistant \
  --dry-run
```

**Deployment Notes**: Integrate with Slack/Teams bot. Low resource requirements. CPU inference sufficient.

**Success Metrics**:
- SOP lookup accuracy: >95%
- Time to find answer: <10 seconds (vs. 5+ minutes manual)
- Adoption rate: >60% of staff using within 3 months

**Security & Compliance**:
- Version control for SOP updates
- Audit trail for procedure lookups
- Access control by department

### Advanced

**Scenario**: Guided workflow automation with multi-step procedures, conditional logic, and integration with ticketing systems.

**Deployment Notes**: Integrate with ServiceNow/Jira for automated ticket creation. Use state machines for multi-step workflows.

**Success Metrics**:
- Workflow completion rate: >85%
- Error reduction in procedures: >50%
- Time savings per workflow: >30%

**Advanced Tips**:
- Implement procedure versioning with automatic model updates
- Use RAG for dynamic procedure lookup based on context
- A/B test workflow prompts for completion optimization

---

## 5. Product Documentation Assistant

Help users navigate product documentation and find relevant information quickly.

### Basic

**Scenario**: A software company wants to help users find answers in their technical documentation.

**CLI Example**:
```bash
foremforge train \
  --data ./data/product_docs_qa.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output models/docs-assistant \
  --dry-run
```

**API Example**:
```python
response = client.post("/predict", json={
    "input": "How do I configure SSO for my organization?",
    "parameters": {"max_tokens": 200}
})
```

**Deployment Notes**: Embed in documentation site. Use client-side inference for privacy. Target <1s response.

**Success Metrics**:
- Documentation search success rate: >80%
- Support ticket reduction: >20%
- User engagement: >3 queries per session

**Security & Compliance**:
- No user query logging without consent
- Implement rate limiting per user
- GDPR-compliant analytics only

### Advanced

**Scenario**: Multi-product documentation with version-aware responses and code example generation.

**Deployment Notes**: Deploy per-version models or use version as context. Integrate with code playgrounds for executable examples.

**Success Metrics**:
- Version-correct response rate: >95%
- Code example validity: >90%
- Developer satisfaction: >4.5/5

**Advanced Tips**:
- Train separate LoRA adapters per product version
- Implement feedback loop for documentation improvements
- Use hybrid search (semantic + keyword) for API reference queries

---

## 6. Compliance & Audit Support

Assist compliance teams with policy interpretation and audit preparation.

### Basic

**Scenario**: Help compliance officers quickly find relevant regulations and internal policy interpretations.

**CLI Example**:
```bash
foremforge train \
  --data ./data/compliance_qa.jsonl \
  --base-model microsoft/phi-2 \
  --output models/compliance-assistant \
  --dry-run
```

**Deployment Notes**: On-premises deployment required. Air-gapped for sensitive environments. Full audit logging mandatory.

**Success Metrics**:
- Policy lookup accuracy: >95%
- Audit preparation time reduction: >40%
- False positive rate in compliance checks: <5%

**Security & Compliance**:
- SOC 2 Type II compliant deployment
- All queries logged with user attribution
- Data encryption at rest and in transit
- Regular access reviews

### Advanced

**Scenario**: Automated compliance checking with evidence gathering and report generation.

**Deployment Notes**: Integrate with GRC platforms (ServiceNow GRC, Archer). Implement approval workflows for high-risk queries.

**Success Metrics**:
- Compliance violation detection rate: >90%
- Report generation time: <1 hour (vs. 8+ hours manual)
- Audit finding accuracy: >95%

**Advanced Tips**:
- Implement regulatory update monitoring with automatic retraining triggers
- Use chain-of-thought prompting for complex compliance reasoning
- A/B test different prompt structures for accuracy optimization

---

## 7. Sales Enablement Assistant

Help sales teams find competitive intelligence, product information, and objection handling responses.

### Basic

**Scenario**: Sales reps need quick access to competitive battlecards and product comparisons.

**CLI Example**:
```bash
foremforge train \
  --data ./data/sales_battlecards.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output models/sales-assistant \
  --dry-run
```

**Deployment Notes**: Mobile-friendly API for field sales. Offline capability for areas with poor connectivity.

**Success Metrics**:
- Rep confidence score: >4/5
- Win rate improvement: >10%
- Time to find information: <30 seconds

**Security & Compliance**:
- Competitor information handling policies
- No storage of customer-specific deal information
- Access logging for audit trails

### Advanced

**Scenario**: Deal-specific recommendations with CRM integration and personalized coaching.

**Deployment Notes**: Integrate with Salesforce/HubSpot. Use deal context for personalized responses. Real-time objection handling during calls.

**Success Metrics**:
- Deal velocity improvement: >15%
- Coaching adoption rate: >70%
- Revenue attribution: trackable per-feature

**Advanced Tips**:
- Train on won deal transcripts for best-practice extraction
- Implement A/B testing for different objection handling approaches
- Use feedback loops from deal outcomes for model improvement

---

## 8. Manufacturing / Industrial SOP Models

Guide technicians through equipment maintenance, troubleshooting, and safety procedures.

### Basic

**Scenario**: Factory technicians need quick access to equipment maintenance procedures on the shop floor.

**CLI Example**:
```bash
foremforge train \
  --data ./data/equipment_sops.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output models/maintenance-assistant \
  --dry-run
```

**Deployment Notes**: Edge deployment on ruggedized tablets. Offline-first architecture. Voice interface integration.

**Success Metrics**:
- Procedure lookup time: <15 seconds
- Maintenance error reduction: >30%
- Equipment downtime reduction: >20%

**Security & Compliance**:
- Safety-critical response validation
- Audit trail for all procedure lookups
- Version control for SOP updates

### Advanced

**Scenario**: Predictive maintenance integration with IoT sensor data and automated work order generation.

**Deployment Notes**: Integrate with SCADA/IoT platforms. Real-time alerts for anomaly detection. GPU inference for image-based diagnostics.

**Success Metrics**:
- Predictive accuracy: >85%
- Unplanned downtime reduction: >40%
- Safety incident reduction: >50%

**Advanced Tips**:
- Combine with computer vision for visual inspection guidance
- Implement escalation paths for safety-critical procedures
- Use federated learning across plants for model improvement

---

## 9. Healthcare (Guideline-only) Assistant

Provide clinical guideline lookups and administrative assistance (non-diagnostic).

### Basic

**Scenario**: Help clinical staff quickly find treatment guidelines and protocol information.

**CLI Example**:
```bash
foremforge train \
  --data ./data/clinical_guidelines.jsonl \
  --base-model microsoft/phi-2 \
  --output models/guideline-assistant \
  --dry-run
```

**Deployment Notes**: HIPAA-compliant infrastructure required. On-premises deployment. No PHI in training data.

**Success Metrics**:
- Guideline retrieval accuracy: >95%
- Time to find protocol: <20 seconds
- Clinician satisfaction: >4/5

**Security & Compliance**:
- HIPAA compliance mandatory
- BAA with infrastructure provider
- PHI redaction in all queries
- Access logging with 7-year retention

### Advanced

**Scenario**: Clinical decision support with EHR integration and care pathway optimization.

**Deployment Notes**: FHIR API integration. Real-time alerts in EHR workflow. HL7 message processing.

**Success Metrics**:
- Alert relevance: >90%
- Care pathway adherence: >85%
- Documentation time reduction: >20%

**Advanced Tips**:
- Implement human-in-the-loop for all clinical recommendations
- Use confidence scoring to gate automated suggestions
- Regular validation against updated clinical guidelines

---

## 10. Legal & Contract Assistance

Help legal teams with contract review, clause identification, and legal research.

### Basic

**Scenario**: Paralegals need to quickly identify key clauses in standard contracts.

**CLI Example**:
```bash
foremforge train \
  --data ./data/contract_clauses.jsonl \
  --base-model microsoft/phi-2 \
  --output models/contract-assistant \
  --dry-run
```

**Deployment Notes**: On-premises for confidentiality. Document upload with automatic processing. Integration with document management systems.

**Success Metrics**:
- Clause identification accuracy: >90%
- Contract review time reduction: >50%
- Risk flag accuracy: >85%

**Security & Compliance**:
- Attorney-client privilege considerations
- Document retention policies
- Access control by matter/client
- Encryption for all stored documents

### Advanced

**Scenario**: Automated contract analysis with risk scoring and negotiation recommendations.

**Deployment Notes**: Integrate with CLM platforms (DocuSign, Ironclad). Workflow automation for approvals.

**Success Metrics**:
- Contract turnaround time: <24 hours
- Risk identification recall: >95%
- Negotiation success rate: >70%

**Advanced Tips**:
- Train on organization-specific playbooks for consistent positions
- Implement version comparison for redline analysis
- Use A/B testing for different clause suggestions

---

## 11. RAG + Tiny Model Hybrid Search

Combine retrieval-augmented generation with fine-tuned models for optimal accuracy.

### Basic

**Scenario**: Enhance model responses with real-time document retrieval for up-to-date information.

**CLI Example**:
```bash
# Build vector index
foremforge index \
  --documents ./data/knowledge_base/ \
  --output ./data/vectors/ \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Query with RAG
foremforge query \
  --model models/knowledge-assistant \
  --index ./data/vectors/ \
  --question "What changed in the Q4 policy update?"
```

**API Example**:
```python
from backend.rag import RAGPipeline

pipeline = RAGPipeline(
    generator_model="models/knowledge-assistant",
    vector_store_path="./data/vectors",
    top_k=5,
    similarity_threshold=0.7
)

result = pipeline.query("What are the new expense limits?")
print(f"Answer: {result['answer']}")
print(f"Sources: {[s['title'] for s in result['sources']]}")
```

**Deployment Notes**: Separate embedding service for scalability. Cache frequent queries. Monitor retrieval latency.

**Success Metrics**:
- Retrieval precision@5: >0.8
- Answer groundedness: >90%
- Source attribution accuracy: >95%

**Security & Compliance**:
- Document-level access control in vector store
- Audit logging for all retrievals
- PII filtering in retrieved content

### Advanced

**Scenario**: Multi-index RAG with cross-domain search and citation generation.

**Deployment Notes**: Implement index routing based on query classification. Use reranking for improved relevance.

**Success Metrics**:
- Cross-domain retrieval accuracy: >85%
- Citation precision: >95%
- Query routing accuracy: >90%

**Advanced Tips**:
- Implement hybrid search (dense + sparse retrieval)
- Use query expansion for improved recall
- A/B test different chunk sizes and overlap strategies

---

## 12. Edge / Offline Deployment

Deploy models on edge devices for offline, low-latency, or air-gapped scenarios.

### Basic

**Scenario**: Field workers need access to procedures without internet connectivity.

**CLI Example**:
```bash
# Export model for edge deployment
foremforge export \
  --model models/sop-assistant \
  --format onnx \
  --quantize int8 \
  --output models/sop-assistant-edge

# Run on edge device
foremforge serve \
  --model models/sop-assistant-edge \
  --port 8080 \
  --device cpu
```

**Deployment Notes**: Target devices: tablets, ruggedized laptops, embedded systems. Memory: 2-4GB minimum. Storage: 500MB-2GB per model.

**Success Metrics**:
- Inference latency: <500ms on CPU
- Model size: <500MB quantized
- Battery impact: <10% per hour of active use

**Security & Compliance**:
- Device encryption mandatory
- Model weight protection
- Offline audit log sync when connected
- Remote wipe capability

### Advanced

**Scenario**: Fleet deployment with OTA updates, telemetry, and federated learning.

**CLI Example**:
```bash
# Package for OTA distribution
foremforge package \
  --model models/sop-assistant-edge \
  --version 1.2.0 \
  --output packages/sop-assistant-1.2.0.tfa

# Deploy to fleet
foremforge deploy \
  --package packages/sop-assistant-1.2.0.tfa \
  --fleet production-tablets \
  --rollout-percentage 10
```

**Deployment Notes**: MDM integration for enterprise deployments. Staged rollouts recommended. Implement model version fallback.

**Success Metrics**:
- Update success rate: >99%
- Rollback time: <5 minutes
- Fleet synchronization: <24 hours

**Advanced Tips**:
- Implement differential updates to reduce bandwidth
- Use federated learning for privacy-preserving model improvement
- A/B test model versions across device segments
- Monitor device-specific performance metrics

---

## Quick Reference: Use Case Selection Matrix

| Use Case | Data Required | Latency Need | Privacy Level | Recommended Start |
|----------|---------------|--------------|---------------|-------------------|
| Knowledge Assistant | Q&A pairs | Medium | High | Basic |
| Support Automation | Tickets + resolutions | Low | Medium | Basic |
| Summarization | Document + summary pairs | Medium | Medium | Basic |
| SOP Automation | Procedure docs | Low | High | Basic |
| Product Docs | Documentation Q&A | Low | Low | Basic |
| Compliance | Policy Q&A | Medium | Very High | Advanced |
| Sales Enablement | Battlecards, transcripts | Low | Medium | Basic |
| Manufacturing | Equipment SOPs | Very Low | High | Advanced |
| Healthcare | Clinical guidelines | Medium | Very High | Advanced |
| Legal | Contracts, clauses | Medium | Very High | Advanced |
| RAG Hybrid | Documents + Q&A | Medium | Varies | Advanced |
| Edge Deployment | Any above | Very Low | Very High | Advanced |

---

## Next Steps

1. Choose a use case that matches your data and requirements
2. Start with the Basic example to validate feasibility
3. Collect user feedback and iterate
4. Progress to Advanced deployment for production scale

For detailed tutorials, see:
- [Customer Support FAQ Bot Tutorial](../wiki/Tutorial-Customer-Support-Bot)
- [Internal Knowledge Search Tutorial](../wiki/Tutorial-Internal-Knowledge-Search)
- [Technical Manual Assistant Tutorial](../wiki/Tutorial-Technical-Manual-Assistant)
