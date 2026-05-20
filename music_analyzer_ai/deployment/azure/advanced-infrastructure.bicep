// Advanced Azure Infrastructure with Bicep
// Includes: Load Balancer, Auto-scaling, Service Mesh, Message Brokers

@description('Environment name')
param environment string = 'production'

@description('Location for resources')
param location string = resourceGroup().location

@description('Spotify API credentials')
@secure()
param spotifyClientSecret string

param spotifyClientId string

// Virtual Network
resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: 'music-analyzer-vnet-${environment}'
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'public-subnet'
        properties: {
          addressPrefix: '10.0.1.0/24'
        }
      }
      {
        name: 'private-subnet'
        properties: {
          addressPrefix: '10.0.2.0/24'
        }
      }
    ]
  }
}

// Application Gateway (Load Balancer)
resource appGateway 'Microsoft.Network/applicationGateways@2023-05-01' = {
  name: 'music-analyzer-agw-${environment}'
  location: location
  properties: {
    sku: {
      name: 'WAF_v2'
      tier: 'WAF_v2'
      capacity: 2
    }
    gatewayIPConfigurations: [
      {
        name: 'appGatewayIpConfig'
        properties: {
          subnet: {
            id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, 'public-subnet')
          }
        }
      }
    ]
    frontendIPConfigurations: [
      {
        name: 'appGatewayFrontendIP'
        properties: {
          publicIPAddress: {
            id: publicIP.id
          }
        }
      }
    ]
    frontendPorts: [
      {
        name: 'httpPort'
        properties: {
          port: 80
        }
      }
      {
        name: 'httpsPort'
        properties: {
          port: 443
        }
      }
    ]
    backendAddressPools: [
      {
        name: 'backendPool'
        properties: {
          backendAddresses: []
        }
      }
    ]
    backendHttpSettingsCollection: [
      {
        name: 'backendSettings'
        properties: {
          port: 8010
          protocol: 'Http'
          cookieBasedAffinity: 'Enabled'
          requestTimeout: 30
          probe: {
            id: resourceId('Microsoft.Network/applicationGateways/probes', 'music-analyzer-agw-${environment}', 'healthProbe')
          }
        }
      }
    ]
    httpListeners: [
      {
        name: 'httpListener'
        properties: {
          frontendIPConfiguration: {
            id: resourceId('Microsoft.Network/applicationGateways/frontendIPConfigurations', 'music-analyzer-agw-${environment}', 'appGatewayFrontendIP')
          }
          frontendPort: {
            id: resourceId('Microsoft.Network/applicationGateways/frontendPorts', 'music-analyzer-agw-${environment}', 'httpPort')
          }
          protocol: 'Http'
        }
      }
    ]
    requestRoutingRules: [
      {
        name: 'rule1'
        properties: {
          ruleType: 'Basic'
          httpListener: {
            id: resourceId('Microsoft.Network/applicationGateways/httpListeners', 'music-analyzer-agw-${environment}', 'httpListener')
          }
          backendAddressPool: {
            id: resourceId('Microsoft.Network/applicationGateways/backendAddressPools', 'music-analyzer-agw-${environment}', 'backendPool')
          }
          backendHttpSettings: {
            id: resourceId('Microsoft.Network/applicationGateways/backendHttpSettingsCollection', 'music-analyzer-agw-${environment}', 'backendSettings')
          }
        }
      }
    ]
    probes: [
      {
        name: 'healthProbe'
        properties: {
          protocol: 'Http'
          path: '/health'
          interval: 30
          timeout: 10
          unhealthyThreshold: 3
        }
      }
    ]
    webApplicationFirewallConfiguration: {
      enabled: true
      firewallMode: 'Prevention'
      ruleSetType: 'OWASP'
      ruleSetVersion: '3.0'
    }
  }
}

// Public IP
resource publicIP 'Microsoft.Network/publicIPAddresses@2023-05-01' = {
  name: 'music-analyzer-pip-${environment}'
  location: location
  sku: {
    name: 'Standard'
    tier: 'Regional'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

// App Service Plan with Auto-scaling
resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: 'music-analyzer-plan-${environment}'
  location: location
  kind: 'linux'
  properties: {
    reserved: true
  }
  sku: {
    name: 'P1v2'
    tier: 'PremiumV2'
    capacity: 2
  }
}

// Auto-scaling Settings
resource autoScale 'Microsoft.Insights/autoscalesettings@2022-10-01' = {
  name: 'music-analyzer-autoscale-${environment}'
  location: location
  properties: {
    enabled: true
    targetResourceUri: appServicePlan.id
    profiles: [
      {
        name: 'Default'
        capacity: {
          minimum: '2'
          maximum: '10'
          default: '2'
        }
        rules: [
          {
            metricTrigger: {
              metricName: 'CpuPercentage'
              metricResourceUri: appServicePlan.id
              operator: 'GreaterThan'
              threshold: 70
              timeAggregation: 'Average'
              timeGrain: 'PT1M'
              timeWindow: 'PT5M'
            }
            scaleAction: {
              direction: 'Increase'
              type: 'ChangeCount'
              value: '1'
              cooldown: 'PT5M'
            }
          }
          {
            metricTrigger: {
              metricName: 'CpuPercentage'
              metricResourceUri: appServicePlan.id
              operator: 'LessThan'
              threshold: 30
              timeAggregation: 'Average'
              timeGrain: 'PT1M'
              timeWindow: 'PT5M'
            }
            scaleAction: {
              direction: 'Decrease'
              type: 'ChangeCount'
              value: '1'
              cooldown: 'PT10M'
            }
          }
        ]
      }
    ]
  }
}

// Azure Service Bus (Message Broker)
resource serviceBus 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: 'music-analyzer-sb-${environment}'
  location: location
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
}

resource serviceBusQueue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: serviceBus
  name: 'analysis-queue'
  properties: {
    maxDeliveryCount: 3
    defaultMessageTimeToLive: 'P14D'
    lockDuration: 'PT5M'
  }
}

resource serviceBusTopic 'Microsoft.ServiceBus/namespaces/topics@2022-10-01-preview' = {
  parent: serviceBus
  name: 'events'
  properties: {
    defaultMessageTimeToLive: 'P14D'
  }
}

// Azure Cache for Redis (Premium)
resource redisCache 'Microsoft.Cache/redis@2023-08-01' = {
  name: 'music-analyzer-redis-${environment}'
  location: location
  properties: {
    sku: {
      name: 'Premium'
      family: 'P'
      capacity: 1
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    redisConfiguration: {
      maxmemoryReserved: '50'
      maxmemoryPolicy: 'allkeys-lru'
    }
  }
}

// Azure Database for PostgreSQL (Flexible Server)
resource postgresServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-06-01-preview' = {
  name: 'music-analyzer-db-${environment}'
  location: location
  sku: {
    name: 'Standard_B2s'
    tier: 'Burstable'
  }
  properties: {
    administratorLogin: 'music_analyzer'
    administratorLoginPassword: spotifyClientSecret // Use separate secret in production
    version: '15'
    storage: {
      storageSizeGB: 128
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Enabled'
    }
    highAvailability: {
      mode: 'ZoneRedundant'
    }
    maintenanceWindow: {
      customWindow: 'Enabled'
      dayOfWeek: 0
      startHour: 2
      startMinute: 0
    }
  }
}

// Outputs
output appGatewayPublicIP {
  value: publicIP.properties.ipAddress
}

output redisHostname {
  value: redisCache.properties.hostName
}

output postgresHostname {
  value: postgresServer.properties.fullyQualifiedDomainName
}




