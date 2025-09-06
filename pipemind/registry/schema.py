from __future__ import annotations
from typing import Dict, List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field, HttpUrl

IOType = Literal[
    "fastq", "fastq_gz",
    "fasta", "fa", "fai", "dict",
    "bam", "bam_bai", "cram", "crai",
    "bed", "bed_gz",
    "vcf", "vcf_gz", "gvcf", "gvcf_gz", "tbi",
    "idx", "gzi",
    "txt", "tsv", "csv", "html", "png", "zip", "json", "report", "directory", "unknown"
]

class IODecl(BaseModel):
    name: str
    io_type: IOType = Field(description="Semantic type of the artifact")
    path_template: Optional[str] = Field(default=None, description="Snakemake-style path template with wildcards")
    description: Optional[str] = None

class ParamDecl(BaseModel):
    name: str
    param_type: Literal["int","float","str","bool","path","enum","json"] = "str"
    required: bool = False
    default: Optional[Union[int, float, str, bool, dict, list]] = None
    description: Optional[str] = None
    enum: Optional[List[str]] = None

class ToolSpec(BaseModel):
    id: str
    name: str
    rule: str
    description: Optional[str] = None
    message: Optional[str] = None
    inputs: List[IODecl]
    outputs: List[IODecl]
    params: List[ParamDecl] = []
    threads: Optional[int] = None
    conda_env: Optional[str] = None
    command: Optional[str] = None
    script: Optional[str] = None
    benchmark: Optional[str] = None
    log_paths: List[str] = []
    # Extended Snakemake compatibility fields
    resources: Dict[str, Any] = Field(default_factory=dict)
    container: Optional[str] = None
    container_engine: Optional[str] = Field(default=None, description="'docker' or 'singularity' if applicable")
    envmodules: List[str] = []
    wrapper: Optional[str] = None
    run_code: Optional[str] = Field(default=None, description="Raw text of run: python block (not executed)")
    group: Optional[str] = None
    priority: Optional[int] = None
    cache: Optional[bool] = None

class ResourceSpec(BaseModel):
    id: str
    name: str
    resource_type: Literal["file","directory","db","service","secret"] = "file"
    uri: str
    description: Optional[str] = None
    access: Literal["public","private","token","oauth","k8s","cluster"] = "public"

class Registry(BaseModel):
    tools: Dict[str, ToolSpec]
    resources: Dict[str, ResourceSpec] = {}

