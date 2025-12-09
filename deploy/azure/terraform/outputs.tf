# TinyForgeAI Azure Terraform Outputs

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "resource_group_location" {
  description = "Resource group location"
  value       = azurerm_resource_group.main.location
}

output "vnet_name" {
  description = "Virtual network name"
  value       = azurerm_virtual_network.main.name
}

output "cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.main.name
}

output "cluster_fqdn" {
  description = "AKS cluster FQDN"
  value       = azurerm_kubernetes_cluster.main.fqdn
}

output "acr_login_server" {
  description = "ACR login server URL"
  value       = azurerm_container_registry.main.login_server
}

output "acr_admin_username" {
  description = "ACR admin username"
  value       = azurerm_container_registry.main.admin_username
  sensitive   = true
}

output "acr_admin_password" {
  description = "ACR admin password"
  value       = azurerm_container_registry.main.admin_password
  sensitive   = true
}

output "storage_account_name" {
  description = "Storage account name for model storage"
  value       = azurerm_storage_account.models.name
}

output "storage_container_name" {
  description = "Storage container name for models"
  value       = azurerm_storage_container.models.name
}

output "workload_identity_client_id" {
  description = "Client ID for workload identity"
  value       = azurerm_user_assigned_identity.tinyforge.client_id
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.main.name} --name ${azurerm_kubernetes_cluster.main.name}"
}

output "acr_login_command" {
  description = "Command to login to ACR"
  value       = "az acr login --name ${azurerm_container_registry.main.name}"
}

output "push_image_commands" {
  description = "Commands to build and push Docker image"
  value       = <<-EOT
    # Login to ACR
    az acr login --name ${azurerm_container_registry.main.name}

    # Build and tag the image
    docker build -t tinyforge-inference:latest .
    docker tag tinyforge-inference:latest ${azurerm_container_registry.main.login_server}/tinyforge-inference:latest

    # Push the image
    docker push ${azurerm_container_registry.main.login_server}/tinyforge-inference:latest
  EOT
}

output "log_analytics_workspace_id" {
  description = "Log Analytics Workspace ID"
  value       = azurerm_log_analytics_workspace.main.id
}
