'use client';

import { ReactNode } from 'react';

export const DynamicKonva = ({ children, ...props }: { children?: ReactNode; [key: string]: any }) => (
  <div className="konva-stage-placeholder border border-gray-300 bg-gray-50 flex items-center justify-center" {...props}>
    <p className="text-gray-500">Canvas functionality temporarily disabled</p>
    {children}
  </div>
);

export const DynamicLayer = ({ children, ...props }: { children?: ReactNode; [key: string]: any }) => (
  <div className="konva-layer-placeholder" {...props}>{children}</div>
);

export const DynamicRect = (props: any) => (
  <div className="w-4 h-4 bg-blue-500 inline-block" {...props} />
);

export const DynamicCircle = (props: any) => (
  <div className="w-4 h-4 bg-red-500 rounded-full inline-block" {...props} />
);

export const DynamicLine = (props: any) => (
  <div className="w-8 h-0.5 bg-gray-500 inline-block" {...props} />
);

export const DynamicText = ({ text, ...props }: { text?: string; [key: string]: any }) => (
  <span className="text-sm" {...props}>{text || 'Text'}</span>
);

export const DynamicImage = (props: any) => (
  <div className="w-8 h-8 bg-gray-300 inline-block" {...props} />
);

export const DynamicGroup = ({ children, ...props }: { children?: ReactNode; [key: string]: any }) => (
  <div className="konva-group-placeholder inline-block" {...props}>{children}</div>
);    