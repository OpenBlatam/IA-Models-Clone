import { UserState, User } from '../../types';

const initialState: UserState = {
  userId: null,
  userData: null,
  isAuthenticated: false,
};

interface UserAction {
  type: string;
  payload?: any;
}

const userReducer = (
  state: UserState = initialState,
  action: UserAction
): UserState => {
  switch (action.type) {
    case 'SET_USER':
      return {
        ...state,
        userId: action.payload.userId as string,
        userData: action.payload.userData as User,
        isAuthenticated: true,
      };
    case 'LOGOUT':
      return {
        ...state,
        userId: null,
        userData: null,
        isAuthenticated: false,
      };
    default:
      return state;
  }
};

export default userReducer;

