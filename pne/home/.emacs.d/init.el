;;; package --- Debian Bookworm init.el

;; Author:  Jeffrey Liu
;; Version: 2023.12.02.19

;;; Commentary:

;;; Code:

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(eval-when-compile
  (require 'use-package))

(use-package package
  :ensure nil
  :config
  (setq package-enable-at-startup nil)
  (add-to-list 'package-archives
    '("melpa" . "https://melpa.org/packages/")
    'append))

(use-package files
  :ensure nil
  :config
  (setq confirm-kill-processes nil)
  (setq create-lockfiles nil)
  (setq make-backup-files nil))

(use-package dired
  :ensure nil
  :commands
  (dired-get-file-for-visit
   dired-find-alternate-file)
  :after
  (ace-window)
  :init
  (defun dired-ace-window-jump ()
    (interactive)
    (let ((value (dired-get-file-for-visit))
          (current-window (selected-window)))
      (cond
       ((> (length (aw-window-list)) 2)
        (aw-switch-to-window (aw-select nil))
        (funcall 'find-file value))
       (t (call-interactively 'dired-find-file-other-window)))
      (unless (eq current-window (selected-window))
        (delete-window current-window))))
  (defun dired-ace-window-show ()
    (interactive)
    (let ((value (dired-get-file-for-visit))
          (current-window (selected-window)))
      (cond
       ((> (length (aw-window-list)) 2)
        (aw-switch-to-window (aw-select nil))
        (funcall 'find-file value))
       (t (call-interactively 'dired-find-file-other-window)))
      (unless (eq current-window (selected-window))
        (aw-flip-window))))
  (defun dired-select-show ()
    (interactive)
    (let ((value (dired-get-file-for-visit)))
      (if (file-directory-p value)
          (dired-find-alternate-file)
        (dired-ace-window-jump))))
  :bind
  ("C-x C-d" . dired)
  (:map dired-mode-map
        ("i" . (lambda() (interactive) (find-alternate-file "..")))
        ("o" . dired-ace-window-jump)
        ("C-o" . dired-ace-window-show)
        ("RET" . dired-select-show))
  :hook
  (dired-after-readin . (lambda()
                          (rename-buffer
                            (generate-new-buffer-name dired-directory))))
  :config
  (setq dired-listing-switches "-aAlhF --group-directories-first")
  (put 'dired-find-alternate-file 'disabled nil))

(use-package sh-script
  :ensure nil
  :delight
  (sh-mode))

(use-package auto-package-update
  :ensure t
  :commands
  (auto-package-update-maybe)
  :init
  (defvar auto-package-update-delete-old-versions t)
  (defvar auto-package-update-hide-results t)
  (auto-package-update-maybe))

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(package-selected-packages
   '(helm-lsp company lsp-mode pyvenv flycheck yasnippet-snippets yasnippet
              ace-window markdown-mode markdown delight which-key helm
              auto-package-update)))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; :commands  defer  - run this package, use defer instead
;; :hook      defer  - other mode hook to this
;; :bind      defer  - binding keys
;; :defer t          - replace commands for defer
;; :init      before - always run, must have a one above
;; :config    after package loading
;; :delight   hide or abbr mode-line name
;; :after     load after some package
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; C-c a    : ace-window jump
;; C-c b    : buffer begin
;; C-c c    : helm lsp code actions
;; C-c d    : helm lsp diagnostics
;; C-c e    : buffer end
;; C-c f    : lsp python formater
;; C-c g    : magit
;; C-c h    : eshell history
;; C-c i    : helm imenu
;; C-c j    : eshell
;; C-c l    ; lsp menu
;; C-c m    : Set Mark
;; C-c n    : yas-new-snippet
;; C-c s    : ace-window swap
;; C-c w c  : clean tabs
;; C-c w f  : flycheck
;; C-c y    : YASnippets
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(use-package delight
  :ensure t)

(use-package material-theme
  :ensure t
  :init
  (load-theme 'material t))

(use-package whitespace
  :ensure t
  :delight
  (global-whitespace-mode)
  :init
  (global-whitespace-mode t)
  :config
  (setq whitespace-style '(face empty tabs lines-tail trailing)))

(use-package helm
  :ensure t
  :bind
  ("M-x" . helm-M-x)
  ("C-c i" . helm-imenu))

(use-package which-key
  :ensure t
  :commands
  (which-key-mode)
  :delight
  (which-key-mode)
  :init
  (which-key-mode))

(use-package ace-window
  :ensure t
  :commands
  (aw-window-list
   aw-switch-to-window
   aw-select
   aw-flip-window)
  :demand t
  :init
  (defvar aw-keys '(?a ?s ?d ?f ?g ?h ?j ?k ?w ?o))
  (defvar aw-scope 'frame)
  (defvar aw-dispatch-always t)
  :bind
  ("C-c a" . ace-window)
  ("C-c s" . ace-swap-window))

(use-package ibuffer
  :ensure t
  :commands
  (ibuffer-current-buffer
   ibuffer-switch-to-saved-filter-groups
   ibuffer-auto-mode)
  :init
  (defun ibuffer-ace-window-jump ()
    (interactive)
    (let ((buf (ibuffer-current-buffer t)))
      (bury-buffer (current-buffer))
      (cond
       ((> (length (aw-window-list)) 2)
        (aw-switch-to-window (aw-select nil))))
      (switch-to-buffer buf)))
  (defun ibuffer-ace-window-show ()
    (interactive)
    (let ((buf (ibuffer-current-buffer t))
          (current-window (selected-window)))
      (bury-buffer (current-buffer))
      (cond
       ((> (length (aw-window-list)) 2)
        (aw-switch-to-window (aw-select nil))))
      (switch-to-buffer buf)
      (unless (eq current-window (selected-window))
        (aw-flip-window))))
  (defun ibuffer-select-jump ()
    (interactive)
    (let ((buf (ibuffer-current-buffer t)))
      (if buf
          (let ((file (buffer-file-name buf))
                (mode (with-current-buffer buf major-mode)))
            (if mode
                (if (eq mode 'dired-mode)
                    (dired file)
                  (ibuffer-ace-window-jump))))
        (message "wrong major mode"))))
  (defvar ibuffer-show-empty-filter-groups nil)
  (defvar ibuffer-saved-filter-groups
    (quote (("Default"
             ("dir" (mode . dired-mode))
             ("py" (mode . python-mode))
             ("md" (mode . markdown-mode))
             ("sh" (mode . sh-mode))
             ("rst" (mode . rst-mode))
             ("pip" (name . "pip.*.txt"))
             ("cfg" (or
                     (mode . org-mode)
                     (name . "init.el")
                     (name . ".zshrc")
                     (name . ".gitignore")))
             ("esh" (mode . eshell-mode))
             ("eww" (mode . eww-mode))
             ("lsp" (or
                     (name . "\\*lsp.*")
                     (name . "\\*pylsp.*")
                     (name . "\\*ts-ls.*")
                     (name . "\\*eslint.*")))))))
  :bind
  ("C-x C-b" . ibuffer)
  ("C-c w p" . previous-buffer)
  ("C-c w n" . next-buffer)
  (:map ibuffer-mode-map
        ("o" . ibuffer-ace-window-jump)
        ("C-o" . ibuffer-ace-window-show)
        ("RET" . ibuffer-select-jump))
  :hook
  (ibuffer-mode . (lambda()
                    (ibuffer-switch-to-saved-filter-groups "Default")
                    (ibuffer-auto-mode 1)))
  :config
  (setq ibuffer-use-other-window t)
  (setq ibuffer-display-summary nil))

(use-package company
  :ensure t
  :delight
  (company-mode)
  :init
  (defvar company-idle-delay 0)
  (defvar company-minimum-prefix-length 1)
  (defvar company-tooltip-limit 20)
  :hook
  (python-mode . company-mode)
  (markdown-mode . company-mode)
  (sh-mode . company-mode)
  (emacs-lisp-mode . company-mode)
  (eshell-mode . company-mode))

(use-package flycheck
  :ensure t
  :bind
  ("C-c w f" . flycheck-list-errors)
  :hook
  (python-mode . flycheck-mode)
  (sh-mode . flycheck-mode)
  (emacs-lisp-mode . flycheck-mode))

(use-package eshell
  :ensure t
  :bind
  ("C-c j" . eshell)
  ("C-c h" . helm-eshell-history))

(use-package markdown-mode
  :ensure t
  :defer t
  :delight
  (markdown-mode "Md"))

(use-package lsp-mode
  :ensure t
  :commands
  (lsp lsp-register-custom-settings)
  :delight
  (lsp-mode)
  :init
  (defvar lsp-keymap-prefix "C-c l")
  (defvar lsp-pylsp-configuration-sources '("ruff"))
  (defvar lsp-pylsp-plugins-flake8-enabled nil)
  (defvar lsp-pylsp-plugins-ruff-enabled t)
  (defvar lsp-pylsp-plugins-jedi-completion-enabled nil)
  (defvar lsp-pylsp-plugins-jedi-completion-include-class-objects nil)
  (defvar lsp-pylsp-plugins-jedi-completion-include-params nil)
  (defvar lsp-pylsp-plugins-jedi-definition-enabled nil)
  (defvar lsp-pylsp-plugins-jedi-definition-follow-builtin-imports nil)
  (defvar lsp-pylsp-plugins-jedi-definition-follow-imports nil)
  (defvar lsp-pylsp-plugins-jedi-hover-enabled nil)
  (defvar lsp-pylsp-plugins-jedi-references-enabled nil)
  (defvar lsp-pylsp-plugins-jedi-signature-help-enabled nil)
  (defvar lsp-pylsp-plugins-jedi-symbols-all-scopes nil)
  (defvar lsp-pylsp-plugins-jedi-symbols-enabled nil)
  (defvar lsp-pylsp-plugins-mccabe-enabled nil)
  (defvar lsp-pylsp-plugins-preload-enabled nil)
  (defvar lsp-pylsp-plugins-pydocstyle-enabled nil)
  (defvar lsp-pylsp-plugins-rope-autoimport-enabled t)
  (defvar lsp-pylsp-plugins-rope-autoimport-memory t)
  (defvar lsp-pylsp-plugins-rope-completion-eager t)
  (defvar lsp-pylsp-plugins-rope-completion-enabled t)
  (defvar lsp-pylsp-rename-backend 'rope)
  (defvar lsp-headerline-breadcrumb-segments '(symbols))
  (defvar lsp-headerline-breadcrumb-icons-enable nil)
  (defvar lsp-modeline-code-actions-segments '(count))
  (defvar lsp-idle-delay 0.500)
  (defvar lsp-log-io nil)
  (defvar lsp-semantic-tokens-enable nil)
  (defvar lsp-modeline-diagnostics-enable nil)
  (defvar lsp-lens-enable nil)
  :bind
  ("C-c f" . lsp-format-buffer)
  :hook
  (python-mode  . lsp)
  (sh-mode . lsp)
  (lsp-mode . lsp-enable-which-key-integration)
  (lsp-mode .(lambda ()
            (custom-set-faces
             '(lsp-headerline-breadcrumb-symbols-face
                ((t (:foreground "yellow")))))))
  (lsp-mode .(lambda ()
            (custom-set-faces
             '(lsp-headerline-breadcrumb-separator-face
               ((t (:foreground "lightblue")))))))
  :config
  (setq read-process-output-max (* 1024 1024))
  (setq gc-cons-threshold 100000000))

(use-package helm-lsp
  :ensure t
  :bind
  ("C-c c" . helm-lsp-code-actions)
  ("C-c d" . helm-lsp-diagnostics))

(use-package yasnippet
  :ensure t
  :commands
  (yas-global-mode yas-reload-all)
  :delight
  (yas-minor-mode)
  :init
  (defvar yas-minor-mode-map)
  (yas-global-mode 1)
  (yas-reload-all)
  :bind
  ("C-c n" . yas-new-snippet)
  (:map yas-minor-mode-map
        ("C-c y" . yas-expand)))
  ;;:hook
  ;;(python-mode . yas-minor-mode)
  ;;(markdown-mode . yas-minor-mode))

(use-package yasnippet-snippets
  :ensure t
  :defer t)

(use-package magit
  :ensure t
  :commands magit-status
  :bind
  ("C-c g" . magit-status))

(use-package pyvenv
  :ensure t
  :commands
  (pyvenv-mode pyvenv-activate)
  :init
  (pyvenv-mode t)
  (pyvenv-activate "/home/pne/env/pne"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(use-package emacs
  :commands
  (uhd)
  :delight
  (visual-line-mode)
  (emacs-lisp-mode "EL")
  (rst-mode "RST")
  (python-mode "Py")
  (lisp-interaction-mode "LI")
  (eldoc-mode)
  :init
  (menu-bar-mode -1)
  (tool-bar-mode -1)
  (electric-pair-mode 1)
  (show-paren-mode 1)
  (winner-mode 1)
  (line-number-mode 1)
  (column-number-mode 1)
  (global-visual-line-mode 1)
  (global-auto-revert-mode 1)
  ;; (global-hl-line-mode 1)
  (defun uhd()
    "init window layout uhd"
    (interactive)
    (split-window-right 82)
    (windmove-right)
    (split-window-below -20)
    (split-window-right 82)
    (windmove-down)
    (split-window-right 82)
    (windmove-left)
    )
  :bind
  ("C-c b" . beginning-of-buffer)
  ("C-c e" . end-of-buffer)
  ("C-c m" . set-mark-command)
  ("C-c w a" . winner-undo)
  ("C-c w e" . winner-redo)
  ("C-c w c" . untabify)
  :hook
  (find-file . read-only-mode)
  :config
  (setq tab-width 2)
  (setq indent-tabs-mode nil)
  (setq inhibit-startup-message t)
  (setq initial-scratch-message ";;; Elisp")
  (setq frame-background-mode 'dark)
  (setq mode-line-frame-identification "")
  (setq-default mode-line-format
                '("%*%l:%C"
                  mode-line-modes
                  "%b"
                  mode-line-misc-info
                  "%-"))
  (setq display-buffer-alist
        '(("^~/$\\|^~/.*/$\\|^/.*/$\\|\\*Flycheck errors\\*\\|\\*Ibuffer\\*"
           (display-buffer-reuse-window
               display-buffer-in-side-window)
           (side            . bottom)
           (reusable-frames . visible)
           (window-height   . 0.30))
          ("\\*compilation\\*"
           display-buffer-no-window
           (allow-no-window . t))))
  (uhd)
  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(provide 'init)
;;; init.el ends here
