import { HistoryState, HistoryItem } from '../../types';

const initialState: HistoryState = {
  history: [],
  timeline: [],
  isLoading: false,
  error: null,
};

interface HistoryAction {
  type: string;
  payload?: any;
}

const historyReducer = (
  state: HistoryState = initialState,
  action: HistoryAction
): HistoryState => {
  switch (action.type) {
    case 'HISTORY_LOAD_START':
      return {
        ...state,
        isLoading: true,
        error: null,
      };
    case 'HISTORY_LOAD_SUCCESS':
      return {
        ...state,
        history: (action.payload as HistoryItem[]) || [],
        isLoading: false,
        error: null,
      };
    case 'HISTORY_LOAD_FAILURE':
      return {
        ...state,
        isLoading: false,
        error: action.payload as string,
      };
    case 'TIMELINE_LOAD_SUCCESS':
      return {
        ...state,
        timeline: action.payload || [],
      };
    case 'ADD_TO_HISTORY':
      return {
        ...state,
        history: [action.payload as HistoryItem, ...state.history],
      };
    default:
      return state;
  }
};

export default historyReducer;

