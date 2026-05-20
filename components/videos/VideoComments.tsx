"use client";
import React, { useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ThumbsUp, MessageSquare, MoreVertical, Flag, ThumbsDown, Reply } from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { useSession, signIn } from "next-auth/react";
import { formatDistanceToNow } from "date-fns";
import { es } from "date-fns/locale";
import { LoadingState } from "@/components/ui/loading-state";
import { useVideoData } from "@/lib/hooks/useVideoData";

interface VideoCommentsProps {
  videoId: string;
  courseId: string;
}

export default function VideoComments({ videoId, courseId }: VideoCommentsProps) {
  const { data: session } = useSession();
  const { data, isLoading, error, addComment } = useVideoData(videoId, courseId);
  const [newComment, setNewComment] = useState("");
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [replyContent, setReplyContent] = useState("");
  const [expandedReplies, setExpandedReplies] = useState<Set<string>>(new Set());

  if (isLoading) {
    return <LoadingState text="Cargando comentarios..." />;
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-destructive">{error}</p>
      </div>
    );
  }

  const comments = data?.comments || [];

  const handleAddComment = async () => {
    if (!session?.user) {
      toast.error("Debes iniciar sesión para comentar");
      return;
    }

    if (!newComment.trim()) return;

    try {
      await addComment(newComment);
      setNewComment("");
      toast.success("Comentario publicado");
    } catch (error) {
      toast.error("Error al publicar el comentario");
    }
  };

  const handleAddReply = async (commentId: string) => {
    if (!session?.user) {
      toast.error("Debes iniciar sesión para responder");
      return;
    }

    if (!replyContent.trim()) return;

    try {
      await addComment(replyContent);
      setReplyContent("");
      setReplyingTo(null);
      toast.success("Respuesta publicada");
    } catch (error) {
      toast.error("Error al publicar la respuesta");
    }
  };

  const handleLike = async (commentId: string) => {
    if (!session?.user) {
      toast.error("Debes iniciar sesión para dar me gusta");
      return;
    }

    try {
      const response = await fetch(`/api/comments?id=${commentId}&action=like&videoId=${videoId}&courseId=${courseId}`, {
        method: "PUT",
      });

      if (!response.ok) throw new Error("Error liking comment");

      const updatedComment = await response.json();
      // Actualizar el comentario en el estado local
      const updatedComments = comments.map((comment) =>
        comment.id === commentId
          ? {
              ...comment,
              likes: updatedComment.likes,
              isLiked: updatedComment.isLiked,
            }
          : comment
      );
    } catch (error) {
      toast.error("Error al dar me gusta");
    }
  };

  const CommentItem = ({ comment, isReply = false }: { comment: any; isReply?: boolean }) => {
    const hasReplies = comment.replies && comment.replies.length > 0;
    const isExpanded = expandedReplies.has(comment.id);

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className={cn(
          "flex gap-4",
          isReply && "ml-8 border-l-2 border-border pl-4"
        )}
      >
        <Avatar className="w-10 h-10">
          <AvatarImage src={comment.user.image} />
          <AvatarFallback>
            {comment.user.name.charAt(0)}
          </AvatarFallback>
        </Avatar>
        <div className="flex-1 space-y-2">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{comment.user.name}</span>
                <span className="text-sm text-muted-foreground">
                  {formatDistanceToNow(new Date(comment.createdAt), {
                    addSuffix: true,
                    locale: es,
                  })}
                </span>
              </div>
              <p className="mt-1 text-sm">{comment.content}</p>
            </div>
            {session?.user && (
              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  <Button variant="ghost" size="icon">
                    <MoreVertical className="w-4 h-4" />
                  </Button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content>
                  <DropdownMenu.Item className="gap-2">
                    <Flag className="w-4 h-4" />
                    Reportar
                  </DropdownMenu.Item>
                </DropdownMenu.Content>
              </DropdownMenu.Root>
            )}
          </div>

          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              className={cn(
                "gap-2",
                comment.isLiked && "text-primary"
              )}
              onClick={() => handleLike(comment.id)}
            >
              <ThumbsUp className="w-4 h-4" />
              {comment.likes}
            </Button>
            {session?.user && !isReply && (
              <Button
                variant="ghost"
                size="sm"
                className="gap-2"
                onClick={() => setReplyingTo(comment.id)}
              >
                <Reply className="w-4 h-4" />
                Responder
              </Button>
            )}
            {hasReplies && !isReply && (
              <Button
                variant="ghost"
                size="sm"
                className="gap-2"
                onClick={() => {
                  setExpandedReplies(prev => {
                    const next = new Set(prev);
                    if (next.has(comment.id)) {
                      next.delete(comment.id);
                    } else {
                      next.add(comment.id);
                    }
                    return next;
                  });
                }}
              >
                <MessageSquare className="w-4 h-4" />
                {isExpanded ? "Ocultar respuestas" : `Ver ${comment.replies?.length} respuestas`}
              </Button>
            )}
          </div>

          {replyingTo === comment.id && !isReply && (
            <div className="mt-4 space-y-2">
              <Textarea
                placeholder="Escribe tu respuesta..."
                value={replyContent}
                onChange={(e) => setReplyContent(e.target.value)}
                className="min-h-[80px]"
              />
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setReplyingTo(null);
                    setReplyContent("");
                  }}
                >
                  Cancelar
                </Button>
                <Button
                  size="sm"
                  onClick={() => handleAddReply(comment.id)}
                >
                  Responder
                </Button>
              </div>
            </div>
          )}

          {isExpanded && hasReplies && (
            <div className="mt-4 space-y-4">
              {comment.replies?.map((reply: any) => (
                <CommentItem key={reply.id} comment={reply} isReply />
              ))}
            </div>
          )}
        </div>
      </motion.div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Comment Input */}
      {session?.user ? (
        <div className="flex gap-4">
          <Avatar className="w-10 h-10">
            <AvatarImage src={session.user.image || undefined} />
            <AvatarFallback>
              {session.user.name?.charAt(0) || "U"}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1 space-y-4">
            <Textarea
              placeholder="Añade un comentario..."
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              className="min-h-[80px] resize-none"
            />
            <div className="flex justify-end gap-2">
              <Button
                variant="ghost"
                onClick={() => setNewComment("")}
                disabled={!newComment.trim()}
              >
                Cancelar
              </Button>
              <Button onClick={handleAddComment} disabled={!newComment.trim()}>
                Comentar
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-4 bg-muted rounded-lg text-center">
          <p className="text-sm text-muted-foreground mb-2">
            Inicia sesión para comentar
          </p>
          <Button variant="default" onClick={() => signIn()}>
            Iniciar sesión
          </Button>
        </div>
      )}

      {/* Comments List */}
      <div className="space-y-6">
        <AnimatePresence>
          {comments.map((comment) => (
            <CommentItem key={comment.id} comment={comment} />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
