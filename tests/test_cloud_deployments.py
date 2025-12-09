"""
Tests for Cloud Deployment Templates

Validates that all cloud deployment templates exist, have correct structure,
and contain required configuration elements.
"""

import os
import pytest
import yaml
import json


# Custom YAML loader for CloudFormation templates
class CloudFormationLoader(yaml.SafeLoader):
    """YAML loader that handles CloudFormation intrinsic functions."""
    pass


# Add constructors for CloudFormation intrinsic functions
def cloudformation_constructor(loader, tag_suffix, node):
    """Generic constructor for CloudFormation tags."""
    if isinstance(node, yaml.ScalarNode):
        return {tag_suffix: loader.construct_scalar(node)}
    elif isinstance(node, yaml.SequenceNode):
        return {tag_suffix: loader.construct_sequence(node)}
    elif isinstance(node, yaml.MappingNode):
        return {tag_suffix: loader.construct_mapping(node)}


# Register CloudFormation intrinsic functions
cf_tags = ['Ref', 'Sub', 'GetAtt', 'Join', 'Select', 'Split', 'If', 'Equals',
           'And', 'Or', 'Not', 'Condition', 'FindInMap', 'Base64', 'Cidr',
           'GetAZs', 'ImportValue', 'Transform']

for tag in cf_tags:
    CloudFormationLoader.add_constructor(
        f'!{tag}',
        lambda loader, node, tag=tag: cloudformation_constructor(loader, tag, node)
    )

# Base path for deploy directory
DEPLOY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deploy")


class TestDeploymentStructure:
    """Test that deployment directory structure is correct."""

    def test_deploy_directory_exists(self):
        """Test deploy directory exists."""
        assert os.path.isdir(DEPLOY_DIR)

    def test_aws_directory_exists(self):
        """Test AWS deployment directory exists."""
        aws_dir = os.path.join(DEPLOY_DIR, "aws")
        assert os.path.isdir(aws_dir)

    def test_gcp_directory_exists(self):
        """Test GCP deployment directory exists."""
        gcp_dir = os.path.join(DEPLOY_DIR, "gcp")
        assert os.path.isdir(gcp_dir)

    def test_azure_directory_exists(self):
        """Test Azure deployment directory exists."""
        azure_dir = os.path.join(DEPLOY_DIR, "azure")
        assert os.path.isdir(azure_dir)

    def test_k8s_directory_exists(self):
        """Test K8s manifests directory exists."""
        k8s_dir = os.path.join(DEPLOY_DIR, "k8s")
        assert os.path.isdir(k8s_dir)

    def test_helm_chart_directory_exists(self):
        """Test Helm chart directory exists."""
        helm_dir = os.path.join(DEPLOY_DIR, "helm-chart")
        assert os.path.isdir(helm_dir)


class TestAWSDeployment:
    """Test AWS deployment templates."""

    @pytest.fixture
    def aws_dir(self):
        return os.path.join(DEPLOY_DIR, "aws")

    def test_cloudformation_template_exists(self, aws_dir):
        """Test CloudFormation template exists."""
        cf_path = os.path.join(aws_dir, "cloudformation-eks.yaml")
        assert os.path.isfile(cf_path)

    def test_cloudformation_template_valid_yaml(self, aws_dir):
        """Test CloudFormation template is valid YAML."""
        cf_path = os.path.join(aws_dir, "cloudformation-eks.yaml")
        with open(cf_path, "r") as f:
            template = yaml.load(f, Loader=CloudFormationLoader)
        assert template is not None

    def test_cloudformation_has_required_sections(self, aws_dir):
        """Test CloudFormation template has required sections."""
        cf_path = os.path.join(aws_dir, "cloudformation-eks.yaml")
        with open(cf_path, "r") as f:
            template = yaml.load(f, Loader=CloudFormationLoader)

        assert "AWSTemplateFormatVersion" in template
        assert "Parameters" in template
        assert "Resources" in template
        assert "Outputs" in template

    def test_cloudformation_has_eks_cluster(self, aws_dir):
        """Test CloudFormation template includes EKS cluster."""
        cf_path = os.path.join(aws_dir, "cloudformation-eks.yaml")
        with open(cf_path, "r") as f:
            template = yaml.load(f, Loader=CloudFormationLoader)

        resources = template.get("Resources", {})
        eks_resources = [
            k for k, v in resources.items()
            if v.get("Type", "").startswith("AWS::EKS::")
        ]
        assert len(eks_resources) > 0

    def test_terraform_directory_exists(self, aws_dir):
        """Test Terraform directory exists."""
        tf_dir = os.path.join(aws_dir, "terraform")
        assert os.path.isdir(tf_dir)

    def test_terraform_main_exists(self, aws_dir):
        """Test Terraform main.tf exists."""
        main_tf = os.path.join(aws_dir, "terraform", "main.tf")
        assert os.path.isfile(main_tf)

    def test_terraform_variables_exists(self, aws_dir):
        """Test Terraform variables.tf exists."""
        vars_tf = os.path.join(aws_dir, "terraform", "variables.tf")
        assert os.path.isfile(vars_tf)

    def test_terraform_outputs_exists(self, aws_dir):
        """Test Terraform outputs.tf exists."""
        outputs_tf = os.path.join(aws_dir, "terraform", "outputs.tf")
        assert os.path.isfile(outputs_tf)

    def test_terraform_example_vars_exists(self, aws_dir):
        """Test Terraform example vars file exists."""
        example = os.path.join(aws_dir, "terraform", "terraform.tfvars.example")
        assert os.path.isfile(example)

    def test_readme_exists(self, aws_dir):
        """Test README exists."""
        readme = os.path.join(aws_dir, "README.md")
        assert os.path.isfile(readme)


class TestGCPDeployment:
    """Test GCP deployment templates."""

    @pytest.fixture
    def gcp_dir(self):
        return os.path.join(DEPLOY_DIR, "gcp")

    def test_terraform_directory_exists(self, gcp_dir):
        """Test Terraform directory exists."""
        tf_dir = os.path.join(gcp_dir, "terraform")
        assert os.path.isdir(tf_dir)

    def test_terraform_main_exists(self, gcp_dir):
        """Test Terraform main.tf exists."""
        main_tf = os.path.join(gcp_dir, "terraform", "main.tf")
        assert os.path.isfile(main_tf)

    def test_terraform_variables_exists(self, gcp_dir):
        """Test Terraform variables.tf exists."""
        vars_tf = os.path.join(gcp_dir, "terraform", "variables.tf")
        assert os.path.isfile(vars_tf)

    def test_terraform_outputs_exists(self, gcp_dir):
        """Test Terraform outputs.tf exists."""
        outputs_tf = os.path.join(gcp_dir, "terraform", "outputs.tf")
        assert os.path.isfile(outputs_tf)

    def test_terraform_example_vars_exists(self, gcp_dir):
        """Test Terraform example vars file exists."""
        example = os.path.join(gcp_dir, "terraform", "terraform.tfvars.example")
        assert os.path.isfile(example)

    def test_readme_exists(self, gcp_dir):
        """Test README exists."""
        readme = os.path.join(gcp_dir, "README.md")
        assert os.path.isfile(readme)

    def test_terraform_has_gke_cluster(self, gcp_dir):
        """Test Terraform includes GKE cluster resource."""
        main_tf = os.path.join(gcp_dir, "terraform", "main.tf")
        with open(main_tf, "r") as f:
            content = f.read()

        assert "google_container_cluster" in content


class TestAzureDeployment:
    """Test Azure deployment templates."""

    @pytest.fixture
    def azure_dir(self):
        return os.path.join(DEPLOY_DIR, "azure")

    def test_terraform_directory_exists(self, azure_dir):
        """Test Terraform directory exists."""
        tf_dir = os.path.join(azure_dir, "terraform")
        assert os.path.isdir(tf_dir)

    def test_terraform_main_exists(self, azure_dir):
        """Test Terraform main.tf exists."""
        main_tf = os.path.join(azure_dir, "terraform", "main.tf")
        assert os.path.isfile(main_tf)

    def test_terraform_variables_exists(self, azure_dir):
        """Test Terraform variables.tf exists."""
        vars_tf = os.path.join(azure_dir, "terraform", "variables.tf")
        assert os.path.isfile(vars_tf)

    def test_terraform_outputs_exists(self, azure_dir):
        """Test Terraform outputs.tf exists."""
        outputs_tf = os.path.join(azure_dir, "terraform", "outputs.tf")
        assert os.path.isfile(outputs_tf)

    def test_terraform_example_vars_exists(self, azure_dir):
        """Test Terraform example vars file exists."""
        example = os.path.join(azure_dir, "terraform", "terraform.tfvars.example")
        assert os.path.isfile(example)

    def test_readme_exists(self, azure_dir):
        """Test README exists."""
        readme = os.path.join(azure_dir, "README.md")
        assert os.path.isfile(readme)

    def test_terraform_has_aks_cluster(self, azure_dir):
        """Test Terraform includes AKS cluster resource."""
        main_tf = os.path.join(azure_dir, "terraform", "main.tf")
        with open(main_tf, "r") as f:
            content = f.read()

        assert "azurerm_kubernetes_cluster" in content


class TestTerraformSyntax:
    """Test Terraform files have valid syntax structure."""

    @pytest.fixture(params=["aws", "gcp", "azure"])
    def cloud_provider(self, request):
        return request.param

    def test_terraform_has_provider_block(self, cloud_provider):
        """Test Terraform has provider configuration."""
        main_tf = os.path.join(DEPLOY_DIR, cloud_provider, "terraform", "main.tf")
        with open(main_tf, "r") as f:
            content = f.read()

        assert "provider" in content

    def test_terraform_has_terraform_block(self, cloud_provider):
        """Test Terraform has terraform configuration block."""
        main_tf = os.path.join(DEPLOY_DIR, cloud_provider, "terraform", "main.tf")
        with open(main_tf, "r") as f:
            content = f.read()

        assert "terraform {" in content

    def test_terraform_has_required_providers(self, cloud_provider):
        """Test Terraform has required_providers block."""
        main_tf = os.path.join(DEPLOY_DIR, cloud_provider, "terraform", "main.tf")
        with open(main_tf, "r") as f:
            content = f.read()

        assert "required_providers" in content

    def test_terraform_has_kubernetes_provider(self, cloud_provider):
        """Test Terraform has Kubernetes provider for Helm deployments."""
        main_tf = os.path.join(DEPLOY_DIR, cloud_provider, "terraform", "main.tf")
        with open(main_tf, "r") as f:
            content = f.read()

        assert 'provider "kubernetes"' in content

    def test_terraform_has_helm_provider(self, cloud_provider):
        """Test Terraform has Helm provider for chart deployment."""
        main_tf = os.path.join(DEPLOY_DIR, cloud_provider, "terraform", "main.tf")
        with open(main_tf, "r") as f:
            content = f.read()

        assert 'provider "helm"' in content


class TestVariableConsistency:
    """Test that variables are consistent across cloud providers."""

    @pytest.fixture(params=["aws", "gcp", "azure"])
    def variables_file(self, request):
        path = os.path.join(DEPLOY_DIR, request.param, "terraform", "variables.tf")
        with open(path, "r") as f:
            return f.read()

    def test_has_environment_variable(self, variables_file):
        """Test environment variable is defined."""
        assert 'variable "environment"' in variables_file

    def test_has_node_min_size(self, variables_file):
        """Test node min size variable is defined."""
        assert "node_min_size" in variables_file

    def test_has_node_max_size(self, variables_file):
        """Test node max size variable is defined."""
        assert "node_max_size" in variables_file

    def test_has_inference_replicas(self, variables_file):
        """Test inference replicas variable is defined."""
        assert "inference_replicas" in variables_file

    def test_has_image_tag(self, variables_file):
        """Test image tag variable is defined."""
        assert "image_tag" in variables_file


class TestReadmeContent:
    """Test README files have required content."""

    @pytest.fixture(params=["aws", "gcp", "azure"])
    def readme_content(self, request):
        path = os.path.join(DEPLOY_DIR, request.param, "README.md")
        with open(path, "r") as f:
            return f.read()

    def test_has_prerequisites(self, readme_content):
        """Test README has prerequisites section."""
        assert "Prerequisites" in readme_content or "prerequisites" in readme_content.lower()

    def test_has_quick_start(self, readme_content):
        """Test README has quick start instructions."""
        assert "Quick Start" in readme_content or "terraform init" in readme_content

    def test_has_cost_section(self, readme_content):
        """Test README has cost information."""
        assert "Cost" in readme_content

    def test_has_cleanup_section(self, readme_content):
        """Test README has cleanup instructions."""
        assert "Cleanup" in readme_content or "destroy" in readme_content


class TestMainDeployReadme:
    """Test main deploy README."""

    @pytest.fixture
    def deploy_readme(self):
        path = os.path.join(DEPLOY_DIR, "README.md")
        with open(path, "r") as f:
            return f.read()

    def test_references_aws(self, deploy_readme):
        """Test main README references AWS."""
        assert "AWS" in deploy_readme

    def test_references_gcp(self, deploy_readme):
        """Test main README references GCP."""
        assert "GCP" in deploy_readme

    def test_references_azure(self, deploy_readme):
        """Test main README references Azure."""
        assert "Azure" in deploy_readme

    def test_has_cloud_guides_section(self, deploy_readme):
        """Test main README has cloud guides section."""
        assert "Cloud Deployment" in deploy_readme or "Cloud-Specific" in deploy_readme
