Example conversation for workflow construction

User: I need filtered SNPs VCF for sample 20AD07372, all lanes.
Assistant: Which input directory and output directory? If not provided, I'll use config.yml defaults. Also need reference panel and DB paths.
User: Use workflow/config.yml as is.
Assistant: Great. I'll parse the workflow, compute the goal artifact `/analysis/006_variant_filtering/{sample}.filtered.snp.vcf`, and fill sample=20AD07372. Lanes are derived by the DAG. Running plan now...
