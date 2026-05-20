import { createStore, combineReducers } from 'redux';
import analysisReducer from './reducers/analysisReducer';
import historyReducer from './reducers/historyReducer';
import userReducer from './reducers/userReducer';
import { RootState } from '../types';

const rootReducer = combineReducers({
  analysis: analysisReducer,
  history: historyReducer,
  user: userReducer,
});

export const store = createStore(rootReducer);

export type AppDispatch = typeof store.dispatch;

