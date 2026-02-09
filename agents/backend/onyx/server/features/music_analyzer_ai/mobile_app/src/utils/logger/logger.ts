import { createLogger } from '../logger-helpers';

export const logger = createLogger('App');

export const apiLogger = createLogger('API');
export const storageLogger = createLogger('Storage');
export const performanceLogger = createLogger('Performance');
export const uiLogger = createLogger('UI');


