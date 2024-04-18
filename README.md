Creating an automated, high-throughput method to extract relevant information from documents, such as peer-reviewed life cycle assessment (LCA) articles and technical reports, is crucial for advancing life cycle inventory (LCI) modeling. Large Language Models (LLMs) can efficiently curate large datasets from various sources, including text descriptions, tabulated data, knowledge graphs, and images. This project aims to create an end-to-end, LLM-based LCI data curation framework. Key steps of this framework include:  
* Detect and partition the key elements (e.g., tables, images) from a given pdf
* Embed and persist the elements into a vector database
* Apply hybrid search to retrieve the relevant information for (1) system boundary completion, (2) inventory data (flow name and quantity) synthesis, (3) assumption validation
* Output the curated LCI data
