import { Resource } from "@/lib/types/academy";
import { FileText, Link, Code } from "lucide-react";

interface ClassResourcesProps {
  resources: Resource[];
}

export function ClassResources({ resources }: ClassResourcesProps) {
  if (!resources || resources.length === 0) {
    return null;
  }

  const getResourceIcon = (type: Resource["type"]) => {
    switch (type) {
      case "pdf":
        return <FileText className="h-5 w-5" />;
      case "link":
        return <Link className="h-5 w-5" />;
      case "code":
        return <Code className="h-5 w-5" />;
      default:
        return <FileText className="h-5 w-5" />;
    }
  };

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold mb-4">Recursos Adicionales</h3>
      <div className="space-y-4">
        {resources.map((resource) => (
          <a
            key={resource.id}
            href={resource.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            {getResourceIcon(resource.type)}
            <div>
              <p className="font-medium">{resource.title}</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {resource.type.toUpperCase()}
              </p>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
} 