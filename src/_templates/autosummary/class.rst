..
  class.rst

{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :special-members: __init__, __getitem__
    :show-inheritance:
    :inherited-members:

    {% block methods %}
        {% if methods %}
            .. rubric:: {{ _('Methods') }}

            .. autosummary::
                :nosignatures:
                {% for item in methods %}
                    ~{{ name }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}