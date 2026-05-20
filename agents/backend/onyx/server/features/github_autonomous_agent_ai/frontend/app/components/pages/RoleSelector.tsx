'use client';

import { motion } from 'framer-motion';

interface Role {
  id: string;
  label: string;
  icon: string;
}

interface RoleSelectorProps {
  roles: Role[];
  selectedRole: string;
  onRoleChange: (roleId: string) => void;
}

export function RoleSelector({ roles, selectedRole, onRoleChange }: RoleSelectorProps) {
  return (
    <div className="flex flex-col items-center mt-16">
      <div className="flex gap-3 md:gap-4">
        {roles.map((role) => (
          <motion.button
            key={role.id}
            onClick={() => onRoleChange(role.id)}
            className={`
              px-6 py-3 rounded-full font-normal text-sm md:text-base transition-all
              focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2
              ${
                selectedRole === role.id
                  ? 'bg-gray-100 text-black border-2 border-black'
                  : 'bg-white text-black border-2 border-gray-300 hover:border-gray-400'
              }
            `}
            aria-label={`Select ${role.label} role`}
            aria-pressed={selectedRole === role.id}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            type="button"
          >
            <div className="flex items-center gap-2">
              <span className="text-base md:text-lg">{role.icon}</span>
              <span>{role.label}</span>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}

