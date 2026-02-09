/**
 * Extended Test Helpers
 * Additional helper functions for testing
 */

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

/**
 * Waits for element to appear and returns it
 */
export async function waitForElement(
  query: () => HTMLElement | null,
  options?: { timeout?: number }
) {
  await waitFor(query, options);
  return query();
}

/**
 * Fills a form field
 */
export async function fillField(
  label: string | RegExp,
  value: string,
  options?: { clear?: boolean }
) {
  const user = userEvent.setup();
  const field = screen.getByLabelText(label);

  if (options?.clear) {
    await user.clear(field);
  }

  await user.type(field, value);
  return field;
}

/**
 * Clicks a button by text
 */
export async function clickButton(text: string | RegExp) {
  const user = userEvent.setup();
  const button = screen.getByRole('button', { name: text });
  await user.click(button);
  return button;
}

/**
 * Selects an option from a select
 */
export async function selectOption(
  label: string | RegExp,
  option: string
) {
  const user = userEvent.setup();
  const select = screen.getByLabelText(label);
  await user.selectOptions(select, option);
  return select;
}

/**
 * Checks a checkbox
 */
export async function checkCheckbox(label: string | RegExp) {
  const user = userEvent.setup();
  const checkbox = screen.getByLabelText(label);
  await user.click(checkbox);
  return checkbox;
}

/**
 * Asserts that element is visible
 */
export function expectVisible(element: HTMLElement | null) {
  expect(element).toBeInTheDocument();
  expect(element).toBeVisible();
}

/**
 * Asserts that element is hidden
 */
export function expectHidden(element: HTMLElement | null) {
  expect(element).not.toBeVisible();
}

/**
 * Asserts that element has focus
 */
export function expectFocused(element: HTMLElement) {
  expect(element).toHaveFocus();
}

/**
 * Asserts that element is disabled
 */
export function expectDisabled(element: HTMLElement) {
  expect(element).toBeDisabled();
}

/**
 * Asserts that element is enabled
 */
export function expectEnabled(element: HTMLElement) {
  expect(element).not.toBeDisabled();
}

