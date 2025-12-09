"""
Tests for TinyForgeAI Kubernetes Manifests

Validates Kubernetes YAML manifests for syntax and structure.
"""

import os
import pytest
import yaml
from pathlib import Path


K8S_DIR = Path(__file__).parent.parent / "deploy" / "k8s"


def load_yaml_files():
    """Load all YAML files from k8s directory."""
    yaml_files = []
    for file in K8S_DIR.glob("*.yaml"):
        if file.name != "kustomization.yaml":
            yaml_files.append(file)
    return yaml_files


class TestK8sManifests:
    """Test Kubernetes manifest validity."""

    @pytest.fixture
    def manifest_files(self):
        """Get list of manifest files."""
        return load_yaml_files()

    def test_k8s_directory_exists(self):
        """Test that k8s directory exists."""
        assert K8S_DIR.exists(), f"K8s directory not found: {K8S_DIR}"

    def test_yaml_files_exist(self, manifest_files):
        """Test that YAML files exist."""
        assert len(manifest_files) > 0, "No YAML files found in k8s directory"

    def test_all_yaml_files_valid(self, manifest_files):
        """Test that all YAML files can be parsed."""
        for yaml_file in manifest_files:
            with open(yaml_file, "r") as f:
                try:
                    docs = list(yaml.safe_load_all(f))
                    assert len(docs) > 0, f"Empty YAML file: {yaml_file}"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {yaml_file}: {e}")

    def test_namespace_manifest(self):
        """Test namespace manifest structure."""
        namespace_file = K8S_DIR / "namespace.yaml"
        assert namespace_file.exists(), "namespace.yaml not found"

        with open(namespace_file) as f:
            docs = list(yaml.safe_load_all(f))
            ns_found = False
            for doc in docs:
                if doc and doc.get("kind") == "Namespace":
                    ns_found = True
                    assert doc["metadata"]["name"] == "tinyforge"
            assert ns_found, "No Namespace resource found"

    def test_deployment_manifests(self):
        """Test deployment manifest structure."""
        deployment_files = [
            "deployment.yaml",
            "dashboard-api.yaml",
            "training-worker.yaml",
        ]

        for filename in deployment_files:
            filepath = K8S_DIR / filename
            assert filepath.exists(), f"{filename} not found"

            with open(filepath) as f:
                docs = list(yaml.safe_load_all(f))
                deployment_found = False
                for doc in docs:
                    if doc and doc.get("kind") == "Deployment":
                        deployment_found = True
                        # Check required fields
                        assert "metadata" in doc
                        assert "spec" in doc
                        assert "template" in doc["spec"]
                        assert "containers" in doc["spec"]["template"]["spec"]

                        # Check namespace
                        assert doc["metadata"].get("namespace") == "tinyforge"

                if filename in ["deployment.yaml", "dashboard-api.yaml", "training-worker.yaml"]:
                    assert deployment_found, f"No Deployment resource found in {filename}"

    def test_service_manifests(self):
        """Test service manifest structure."""
        service_files = ["service.yaml", "dashboard-api.yaml"]

        for filename in service_files:
            filepath = K8S_DIR / filename
            with open(filepath) as f:
                docs = list(yaml.safe_load_all(f))
                service_found = False
                for doc in docs:
                    if doc and doc.get("kind") == "Service":
                        service_found = True
                        assert "metadata" in doc
                        assert "spec" in doc
                        assert "ports" in doc["spec"]
                        assert doc["metadata"].get("namespace") == "tinyforge"
                assert service_found, f"No Service resource found in {filename}"

    def test_pvc_manifest(self):
        """Test persistent volume claim manifest."""
        pvc_file = K8S_DIR / "pvc.yaml"
        assert pvc_file.exists(), "pvc.yaml not found"

        with open(pvc_file) as f:
            docs = list(yaml.safe_load_all(f))
            pvc_names = []
            for doc in docs:
                if doc and doc.get("kind") == "PersistentVolumeClaim":
                    pvc_names.append(doc["metadata"]["name"])
                    assert "spec" in doc
                    assert "accessModes" in doc["spec"]
                    assert "resources" in doc["spec"]
                    assert doc["metadata"].get("namespace") == "tinyforge"

            # Check required PVCs exist
            expected_pvcs = ["tinyforge-models-pvc", "tinyforge-data-pvc", "tinyforge-cache-pvc"]
            for pvc in expected_pvcs:
                assert pvc in pvc_names, f"PVC {pvc} not found"

    def test_configmap_manifest(self):
        """Test configmap manifest structure."""
        configmap_file = K8S_DIR / "configmap.yaml"
        assert configmap_file.exists(), "configmap.yaml not found"

        with open(configmap_file) as f:
            docs = list(yaml.safe_load_all(f))
            configmap_found = False
            for doc in docs:
                if doc and doc.get("kind") == "ConfigMap":
                    configmap_found = True
                    assert "data" in doc
                    assert doc["metadata"].get("namespace") == "tinyforge"
            assert configmap_found, "No ConfigMap resource found"

    def test_secrets_manifest(self):
        """Test secrets manifest structure."""
        secrets_file = K8S_DIR / "secrets.yaml"
        assert secrets_file.exists(), "secrets.yaml not found"

        with open(secrets_file) as f:
            docs = list(yaml.safe_load_all(f))
            secret_found = False
            for doc in docs:
                if doc and doc.get("kind") == "Secret":
                    secret_found = True
                    assert "stringData" in doc or "data" in doc
                    assert doc["metadata"].get("namespace") == "tinyforge"
            assert secret_found, "No Secret resource found"

    def test_hpa_manifest(self):
        """Test horizontal pod autoscaler manifest."""
        hpa_file = K8S_DIR / "hpa.yaml"
        assert hpa_file.exists(), "hpa.yaml not found"

        with open(hpa_file) as f:
            docs = list(yaml.safe_load_all(f))
            hpa_count = 0
            for doc in docs:
                if doc and doc.get("kind") == "HorizontalPodAutoscaler":
                    hpa_count += 1
                    assert "spec" in doc
                    assert "scaleTargetRef" in doc["spec"]
                    assert "metrics" in doc["spec"]
                    assert doc["metadata"].get("namespace") == "tinyforge"
            assert hpa_count >= 1, "No HPA resource found"

    def test_ingress_manifest(self):
        """Test ingress manifest structure."""
        ingress_file = K8S_DIR / "ingress.yaml"
        assert ingress_file.exists(), "ingress.yaml not found"

        with open(ingress_file) as f:
            docs = list(yaml.safe_load_all(f))
            ingress_found = False
            for doc in docs:
                if doc and doc.get("kind") == "Ingress":
                    ingress_found = True
                    assert "spec" in doc
                    assert "rules" in doc["spec"]
                    assert doc["metadata"].get("namespace") == "tinyforge"
            assert ingress_found, "No Ingress resource found"

    def test_kustomization_manifest(self):
        """Test kustomization file structure."""
        kustomization_file = K8S_DIR / "kustomization.yaml"
        assert kustomization_file.exists(), "kustomization.yaml not found"

        with open(kustomization_file) as f:
            doc = yaml.safe_load(f)
            assert doc["apiVersion"] == "kustomize.config.k8s.io/v1beta1"
            assert doc["kind"] == "Kustomization"
            assert "resources" in doc
            assert len(doc["resources"]) > 0

    def test_labels_consistency(self, manifest_files):
        """Test that all resources have consistent labels."""
        for yaml_file in manifest_files:
            with open(yaml_file) as f:
                docs = list(yaml.safe_load_all(f))
                for doc in docs:
                    if doc and "metadata" in doc and "labels" in doc["metadata"]:
                        labels = doc["metadata"]["labels"]
                        # All resources should have app=tinyforge label
                        if doc.get("kind") not in ["ResourceQuota"]:
                            assert labels.get("app") == "tinyforge", \
                                f"Missing app=tinyforge label in {yaml_file}"

    def test_resource_requests_and_limits(self):
        """Test that deployments have resource requests and limits."""
        deployment_files = [
            "deployment.yaml",
            "dashboard-api.yaml",
            "training-worker.yaml",
        ]

        for filename in deployment_files:
            filepath = K8S_DIR / filename
            with open(filepath) as f:
                docs = list(yaml.safe_load_all(f))
                for doc in docs:
                    if doc and doc.get("kind") == "Deployment":
                        containers = doc["spec"]["template"]["spec"]["containers"]
                        for container in containers:
                            assert "resources" in container, \
                                f"Container {container['name']} in {filename} missing resources"
                            resources = container["resources"]
                            assert "requests" in resources, \
                                f"Container {container['name']} in {filename} missing requests"
                            assert "limits" in resources, \
                                f"Container {container['name']} in {filename} missing limits"
