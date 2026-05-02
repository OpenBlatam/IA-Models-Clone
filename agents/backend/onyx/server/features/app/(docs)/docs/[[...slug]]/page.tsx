import { notFound } from "next/navigation";

import { getTableOfContents } from "@/lib/toc";
import { Mdx } from "@/components/content/mdx-components";
import { DocsPageHeader } from "@/components/docs/page-header";
import { DocsPager } from "@/components/docs/pager";
import { DashboardTableOfContents } from "@/components/shared/toc";

import "@/styles/mdx.css";

import { Metadata } from "next";

import { constructMetadata, getBlurDataURL } from "@/lib/utils";

interface DocPageProps {
  params: Promise<{
    slug: string[];
  }>;
}

async function getDocFromParams(params: Promise<{ slug: string[] }>) {
  const resolvedParams = await params;
  const slug = resolvedParams.slug?.join("/") || "";
  
  const docs = {
    "": {
      title: "Documentación",
      description: "Documentación completa de Blatam Academy",
      body: { raw: "# Bienvenido a la Documentación\n\nAquí encontrarás toda la información necesaria...", code: "" },
      images: []
    },
    "conceptos-basicos": {
      title: "Conceptos Básicos",
      description: "Aprende los conceptos fundamentales",
      body: { raw: "# Conceptos Básicos\n\nEsta sección cubre los conceptos fundamentales...", code: "" },
      images: []
    },
    "configuration/authentification": {
      title: "Configuración de Autenticación",
      description: "Configura la autenticación en tu aplicación",
      body: { raw: "# Configuración de Autenticación\n\nAprende a configurar la autenticación...", code: "" },
      images: []
    },
    "configuration/blog": {
      title: "Configuración del Blog",
      description: "Configura el sistema de blog",
      body: { raw: "# Configuración del Blog\n\nAprende a configurar el sistema de blog...", code: "" },
      images: []
    },
    "configuration/config-files": {
      title: "Archivos de Configuración",
      description: "Gestiona los archivos de configuración",
      body: { raw: "# Archivos de Configuración\n\nAprende a gestionar los archivos de configuración...", code: "" },
      images: []
    }
  };
  
  const doc = docs[slug as keyof typeof docs];
  if (!doc) return null;

  return {
    ...doc,
    slug: slug,
    slugAsParams: slug
  };
}

export async function generateMetadata({
  params,
}: DocPageProps): Promise<Metadata> {
  const doc = await getDocFromParams(params);

  if (!doc) return {};

  const { title, description } = doc;

  return constructMetadata({
    title: `${title} – SaaS Starter`,
    description: description,
  });
}

export async function generateStaticParams(): Promise<
  { slug: string[] }[]
> {
  return [
    { slug: [] },
    { slug: ["conceptos-basicos"] },
    { slug: ["configuration", "authentification"] },
    { slug: ["configuration", "blog"] },
    { slug: ["configuration", "config-files"] }
  ];
}

export default async function DocPage({ params }: DocPageProps) {
  const doc = await getDocFromParams(params);

  if (!doc) {
    notFound();
  }

  const toc = await getTableOfContents(doc.body.raw);

  const images = await Promise.all(
    doc.images.map(async (src: string) => ({
      src,
      alt: "Documentation image",
      blurDataURL: await getBlurDataURL(src),
    })),
  );

  return (
    <main className="relative py-6 lg:gap-10 lg:py-8 xl:grid xl:grid-cols-[1fr_300px]">
      <div className="mx-auto w-full min-w-0">
        <DocsPageHeader heading={doc.title} text={doc.description} />
        <div className="pb-4 pt-11">
          <Mdx content={doc.body.raw} images={images} />
        </div>
        <hr className="my-4 md:my-6" />
        <DocsPager doc={doc} />
      </div>
      <div className="hidden text-sm xl:block">
        <div className="sticky top-16 -mt-10 max-h-[calc(var(--vh)-4rem)] overflow-y-auto pt-8">
          <DashboardTableOfContents toc={toc} />
        </div>
      </div>
    </main>
  );
}
