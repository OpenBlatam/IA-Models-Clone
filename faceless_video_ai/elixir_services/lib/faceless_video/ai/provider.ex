defmodule FacelessVideo.AI.Provider do
  @moduledoc """
  Behaviour for AI provider implementations.
  """

  @doc """
  Check if the provider is available (has API key configured).
  """
  @callback available?() :: boolean()
end




