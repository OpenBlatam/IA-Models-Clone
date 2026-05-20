"""Registry mapping for connector classes."""

from pydantic import BaseModel

from unified_core.configs.constants import DocumentSource


class ConnectorMapping(BaseModel):
    module_path: str
    class_name: str


# Mapping of DocumentSource to connector details for lazy loading
CONNECTOR_CLASS_MAP = {
    DocumentSource.WEB: ConnectorMapping(
        module_path="unified_core.connectors.web.connector",
        class_name="WebConnector",
    ),
    DocumentSource.FILE: ConnectorMapping(
        module_path="unified_core.connectors.file.connector",
        class_name="LocalFileConnector",
    ),
    DocumentSource.SLACK: ConnectorMapping(
        module_path="unified_core.connectors.slack.connector",
        class_name="SlackConnector",
    ),
    DocumentSource.GITHUB: ConnectorMapping(
        module_path="unified_core.connectors.github.connector",
        class_name="GithubConnector",
    ),
    DocumentSource.GMAIL: ConnectorMapping(
        module_path="unified_core.connectors.gmail.connector",
        class_name="GmailConnector",
    ),
    DocumentSource.GITLAB: ConnectorMapping(
        module_path="unified_core.connectors.gitlab.connector",
        class_name="GitlabConnector",
    ),
    DocumentSource.GITBOOK: ConnectorMapping(
        module_path="unified_core.connectors.gitbook.connector",
        class_name="GitbookConnector",
    ),
    DocumentSource.GOOGLE_DRIVE: ConnectorMapping(
        module_path="unified_core.connectors.google_drive.connector",
        class_name="GoogleDriveConnector",
    ),
    DocumentSource.BOOKSTACK: ConnectorMapping(
        module_path="unified_core.connectors.bookstack.connector",
        class_name="BookstackConnector",
    ),
    DocumentSource.OUTLINE: ConnectorMapping(
        module_path="unified_core.connectors.outline.connector",
        class_name="OutlineConnector",
    ),
    DocumentSource.CONFLUENCE: ConnectorMapping(
        module_path="unified_core.connectors.confluence.connector",
        class_name="ConfluenceConnector",
    ),
    DocumentSource.JIRA: ConnectorMapping(
        module_path="unified_core.connectors.jira.connector",
        class_name="JiraConnector",
    ),
    DocumentSource.PRODUCTBOARD: ConnectorMapping(
        module_path="unified_core.connectors.productboard.connector",
        class_name="ProductboardConnector",
    ),
    DocumentSource.SLAB: ConnectorMapping(
        module_path="unified_core.connectors.slab.connector",
        class_name="SlabConnector",
    ),
    DocumentSource.NOTION: ConnectorMapping(
        module_path="unified_core.connectors.notion.connector",
        class_name="NotionConnector",
    ),
    DocumentSource.ZULIP: ConnectorMapping(
        module_path="unified_core.connectors.zulip.connector",
        class_name="ZulipConnector",
    ),
    DocumentSource.GURU: ConnectorMapping(
        module_path="unified_core.connectors.guru.connector",
        class_name="GuruConnector",
    ),
    DocumentSource.LINEAR: ConnectorMapping(
        module_path="unified_core.connectors.linear.connector",
        class_name="LinearConnector",
    ),
    DocumentSource.HUBSPOT: ConnectorMapping(
        module_path="unified_core.connectors.hubspot.connector",
        class_name="HubSpotConnector",
    ),
    DocumentSource.DOCUMENT360: ConnectorMapping(
        module_path="unified_core.connectors.document360.connector",
        class_name="Document360Connector",
    ),
    DocumentSource.GONG: ConnectorMapping(
        module_path="unified_core.connectors.gong.connector",
        class_name="GongConnector",
    ),
    DocumentSource.GOOGLE_SITES: ConnectorMapping(
        module_path="unified_core.connectors.google_site.connector",
        class_name="GoogleSitesConnector",
    ),
    DocumentSource.ZENDESK: ConnectorMapping(
        module_path="unified_core.connectors.zendesk.connector",
        class_name="ZendeskConnector",
    ),
    DocumentSource.LOOPIO: ConnectorMapping(
        module_path="unified_core.connectors.loopio.connector",
        class_name="LoopioConnector",
    ),
    DocumentSource.DROPBOX: ConnectorMapping(
        module_path="unified_core.connectors.dropbox.connector",
        class_name="DropboxConnector",
    ),
    DocumentSource.SHAREPOINT: ConnectorMapping(
        module_path="unified_core.connectors.sharepoint.connector",
        class_name="SharepointConnector",
    ),
    DocumentSource.TEAMS: ConnectorMapping(
        module_path="unified_core.connectors.teams.connector",
        class_name="TeamsConnector",
    ),
    DocumentSource.SALESFORCE: ConnectorMapping(
        module_path="unified_core.connectors.salesforce.connector",
        class_name="SalesforceConnector",
    ),
    DocumentSource.DISCOURSE: ConnectorMapping(
        module_path="unified_core.connectors.discourse.connector",
        class_name="DiscourseConnector",
    ),
    DocumentSource.AXERO: ConnectorMapping(
        module_path="unified_core.connectors.axero.connector",
        class_name="AxeroConnector",
    ),
    DocumentSource.CLICKUP: ConnectorMapping(
        module_path="unified_core.connectors.clickup.connector",
        class_name="ClickupConnector",
    ),
    DocumentSource.MEDIAWIKI: ConnectorMapping(
        module_path="unified_core.connectors.mediawiki.wiki",
        class_name="MediaWikiConnector",
    ),
    DocumentSource.WIKIPEDIA: ConnectorMapping(
        module_path="unified_core.connectors.wikipedia.connector",
        class_name="WikipediaConnector",
    ),
    DocumentSource.ASANA: ConnectorMapping(
        module_path="unified_core.connectors.asana.connector",
        class_name="AsanaConnector",
    ),
    DocumentSource.S3: ConnectorMapping(
        module_path="unified_core.connectors.blob.connector",
        class_name="BlobStorageConnector",
    ),
    DocumentSource.R2: ConnectorMapping(
        module_path="unified_core.connectors.blob.connector",
        class_name="BlobStorageConnector",
    ),
    DocumentSource.GOOGLE_CLOUD_STORAGE: ConnectorMapping(
        module_path="unified_core.connectors.blob.connector",
        class_name="BlobStorageConnector",
    ),
    DocumentSource.OCI_STORAGE: ConnectorMapping(
        module_path="unified_core.connectors.blob.connector",
        class_name="BlobStorageConnector",
    ),
    DocumentSource.XENFORO: ConnectorMapping(
        module_path="unified_core.connectors.xenforo.connector",
        class_name="XenforoConnector",
    ),
    DocumentSource.DISCORD: ConnectorMapping(
        module_path="unified_core.connectors.discord.connector",
        class_name="DiscordConnector",
    ),
    DocumentSource.FRESHDESK: ConnectorMapping(
        module_path="unified_core.connectors.freshdesk.connector",
        class_name="FreshdeskConnector",
    ),
    DocumentSource.FIREFLIES: ConnectorMapping(
        module_path="unified_core.connectors.fireflies.connector",
        class_name="FirefliesConnector",
    ),
    DocumentSource.EGNYTE: ConnectorMapping(
        module_path="unified_core.connectors.egnyte.connector",
        class_name="EgnyteConnector",
    ),
    DocumentSource.AIRTABLE: ConnectorMapping(
        module_path="unified_core.connectors.airtable.airtable_connector",
        class_name="AirtableConnector",
    ),
    DocumentSource.HIGHSPOT: ConnectorMapping(
        module_path="unified_core.connectors.highspot.connector",
        class_name="HighspotConnector",
    ),
    DocumentSource.IMAP: ConnectorMapping(
        module_path="unified_core.connectors.imap.connector",
        class_name="ImapConnector",
    ),
    DocumentSource.BITBUCKET: ConnectorMapping(
        module_path="unified_core.connectors.bitbucket.connector",
        class_name="BitbucketConnector",
    ),
    # just for integration tests
    DocumentSource.MOCK_CONNECTOR: ConnectorMapping(
        module_path="unified_core.connectors.mock_connector.connector",
        class_name="MockConnector",
    ),
}
