;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; init.el --- Emacs init file for Ubuntu
;;; Author: Jeffrey Liu
;;; Commentary:
;;; Version 2023.05.23.006
;;; Code:
;;; global-set-key "C-c"
;;; a: ace-window
;;; s: ace-swap-window
;;; b: buffer-begin
;;; e: buffer-end
;;; @: hs-minor-mode hide&show block, level and all
;;; C-f: json buffer format
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(require 'package)
(add-to-list 'package-archives
             '("melpa" . "https://melpa.org/packages/")
             'append)
(setq package-enable-at-startup nil)
(package-initialize)
(unless (package-installed-p 'use-package)
        (package-refresh-contents)
        (package-install 'use-package))
(eval-and-compile
  (setq use-package-always-ensure t
        use-package-expand-minimally t))
(use-package auto-package-update
  :init
  (setq auto-package-update-delete-old-versions t)
  (setq auto-package-update-hide-results t)
  :config
  (auto-package-update-maybe))
(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(column-number-mode t)
 '(package-selected-packages
   '(json-reformat json-mode flycheck company which-key use-package helm auto-package-update ace-window))
 '(size-indication-mode t)
 '(tool-bar-mode nil)
 '(tooltip-mode nil))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:family "DejaVu Sans Mono" :foundry "PfEd" :slant normal :weight normal :height 132 :width normal)))))

(use-package ace-window
  :ensure t
  :init
  (setq aw-keys '(?a ?s ?d ?f ?g ?h ?j ?k ?w ?o))
  (setq aw-scope 'frame)
  (setq aw-dispatch-always t)
  :bind
  (("C-c a" . ace-window)
  ("C-c s" . ace-swap-window)))

(use-package helm
  :ensure t
  :bind
  (("M-x" . helm-M-x)))

(use-package which-key
  :ensure t
  :diminish
  :config
  (which-key-mode))

(use-package files
  :ensure nil
  :config
  (setq confirm-kill-processes nil)
  (setq create-lockfiles nil)
  (setq make-backup-files nil))

(use-package company
  :ensure t
  :diminish company-mode
  :init
  (setq company-idle-delay            0
        company-minimum-prefix-length 1
        company-tooltip-limit         20
        company-dabbrev-downcase      nil)
  :bind ("C-c c" . company-mode))

(use-package flycheck
  :ensure t)

(use-package json-mode
  :ensure t
  :config
  (add-hook 'json-mode-hook 'flycheck-mode)
  (add-hook 'json-mode-hook 'hs-minor-mode))

(use-package json-reformat
  :ensure t)

(use-package emacs
  :init
  (setq-default indent-tabs-mode nil)
  (setq line-number-mode t)
  (setq column-number-mode t)
  (setq inhibit-startup-message t)
  (setq initial-scratch-message ";;; Elisp")
  (setq frame-background-mode 'dark)
  (setq select-enable-clipboard-default t)
  (setq select-enable-primary t)
  (set-frame-parameter (selected-frame) 'alpha '(100 . 100))
  (add-to-list 'default-frame-alist '(alpha . (100 . 100))) 
  :bind
  (("C-c b" . beginning-of-buffer)
   ("C-c e" . end-of-buffer)
   ("C-c m" . set-mark-command))
  :hook
  (find-file . read-only-mode)
  :config
  (menu-bar-mode -1)
  (tool-bar-mode -1)
  (scroll-bar-mode -1)
  (electric-pair-mode 1)
  (global-visual-line-mode 1)
  (winner-mode 1)
  (global-auto-revert-mode 1)
  (delete-selection-mode 1)
  (show-paren-mode 1))
