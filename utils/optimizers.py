import tensorflow as tf

def Adam_optimizer_poly(loss,
                        variables,
                        decay_power,
                        starter_learning_rate,
                        end_learning_rate,
                        start_decay_step,
                        decay_steps,
                        beta1,
                        global_step,
                        name_optimizer='Adam'):


    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                      decay_steps, end_learning_rate,
                                      power=decay_power),
            starter_learning_rate
        )

    )

    optimizer_summary_list =[tf.summary.scalar('global_step/{}'.format(name_optimizer), global_step),
                             tf.summary.scalar('learning_rate/{}'.format(name_optimizer), learning_rate)]
    optimizer_summary = tf.summary.merge(optimizer_summary_list)

    learning_step = (
        tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name_optimizer)
            .minimize(loss, var_list=variables)
    )
    return learning_step, optimizer_summary

def Momentum_optimizer_poly(loss,
                           variables,
                           decay_power,
                           starter_learning_rate,
                           end_learning_rate,
                           start_decay_step,
                           decay_steps,
                           momentum,
                           global_step,
                           name_optimizer='Momentum'):

    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                      decay_steps, end_learning_rate,
                                      power=decay_power),
            starter_learning_rate
        )

    )

    optimizer_summary_list = [tf.summary.scalar('global_step/{}'.format(name_optimizer), global_step),
                              tf.summary.scalar('learning_rate/{}'.format(name_optimizer), learning_rate)]
    optimizer_summary =tf.summary.merge(optimizer_summary_list)

    learning_step = (
        tf.train.MomentumOptimizer(learning_rate, momentum=momentum, name=name_optimizer)
            .minimize(loss, var_list=variables)
    )
    return learning_step, optimizer_summary

def RMSProp_optimizer_poly(loss,
                           variables,
                           decay_power,
                           starter_learning_rate,
                           end_learning_rate,
                           start_decay_step,
                           decay_steps,
                           decay,
                           momentum,
                           global_step,
                           name_optimizer='RMSProp'):
    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                      decay_steps, end_learning_rate,
                                      power=decay_power),
            starter_learning_rate
        )

    )

    optimizer_summary_list = [tf.summary.scalar('global_step/{}'.format(name_optimizer), global_step),
                              tf.summary.scalar('learning_rate/{}'.format(name_optimizer), learning_rate)]
    optimizer_summary = tf.summary.merge(optimizer_summary_list)

    learning_step = (
        tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, name=name_optimizer)
            .minimize(loss, var_list=variables)
    )
    return learning_step, optimizer_summary

def RMSProp_separated_optimizer_poly(loss,
                                     variables,
                                     last_layers,
                                     last_layer_gradient_multiplier,
                                     decay_power,
                                     starter_learning_rate,
                                     end_learning_rate,
                                     start_decay_step,
                                     decay_steps,
                                     decay,
                                     momentum,
                                     global_step,
                                     name_optimizer='RMSProp_separated'):
    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                      decay_steps, end_learning_rate,
                                      power=decay_power),
            starter_learning_rate
        )

    )

    tf.summary.scalar('global_step/{}'.format(name_optimizer), global_step)
    tf.summary.scalar('learning_rate/{}'.format(name_optimizer), learning_rate)
    optimizer_summary_list = [tf.summary.scalar('global_step/{}'.format(name_optimizer), global_step),
                              tf.summary.scalar('learning_rate/{}'.format(name_optimizer), learning_rate)]
    optimizer_summary = tf.summary.merge(optimizer_summary_list)

    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, name=name_optimizer)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=variables)

    optimizable_vars = []
    grads_and_vars_without_none_gradients = []
    for grad_and_var in grads_and_vars:
        if grad_and_var[0] is not None:
            optimizable_vars.append(grad_and_var[1])
            grads_and_vars_without_none_gradients.append(grad_and_var)

    grad_mult = _get_model_gradient_multipliers(variables=optimizable_vars, last_layers=last_layers,
                                                last_layer_gradient_multiplier=last_layer_gradient_multiplier)

    grads_and_vars_without_none_gradients = tf.contrib.training.multiply_gradients(
        grads_and_vars_without_none_gradients, grad_mult)
    learning_step = optimizer.apply_gradients(grads_and_vars_without_none_gradients)

    return learning_step, optimizer_summary

def Adam_optimizer_separated_poly(loss,
                                  variables,
                                  last_layers,
                                  last_layer_gradient_multiplier,
                                  decay_power,
                                  starter_learning_rate,
                                  end_learning_rate,
                                  start_decay_step,
                                  decay_steps,
                                  beta1,
                                  global_step,
                                  name_optimizer='Adam'):


    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                      decay_steps, end_learning_rate,
                                      power=decay_power),
            starter_learning_rate
        )

    )

    optimizer_summary_list = [tf.summary.scalar('global_step/{}'.format(name_optimizer), global_step),
                              tf.summary.scalar('learning_rate/{}'.format(name_optimizer), learning_rate)]
    optimizer_summary = tf.summary.merge(optimizer_summary_list)

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name_optimizer)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=variables)

    optimizable_vars = []
    grads_and_vars_without_none_gradients = []
    for grad_and_var in grads_and_vars:
        if grad_and_var[0] is not None:
            optimizable_vars.append(grad_and_var[1])
            grads_and_vars_without_none_gradients.append(grad_and_var)

    grad_mult = _get_model_gradient_multipliers(variables=optimizable_vars, last_layers=last_layers, last_layer_gradient_multiplier=last_layer_gradient_multiplier)


    grads_and_vars_without_none_gradients = tf.contrib.training.multiply_gradients(grads_and_vars_without_none_gradients, grad_mult)
    learning_step = optimizer.apply_gradients(grads_and_vars_without_none_gradients)

    return learning_step, optimizer_summary


def _get_model_gradient_multipliers(variables, last_layers, last_layer_gradient_multiplier):
    """Gets the gradient multipliers.

    The gradient multipliers will adjust the learning rates for model
    variables. For the task of semantic segmentation, the models are
    usually fine-tuned from the models trained on the task of image
    classification. To fine-tune the models, we usually set larger (e.g.,
    10 times larger) learning rate for the parameters of last layer.

    Args:
        last_layers: Scopes of last layers.
        last_layer_gradient_multiplier: The gradient multiplier for last layers.

    Returns:
        The gradient multiplier map with variables as key, and multipliers as value.
    """
    gradient_multipliers = {}

    for var in variables:
        # Double the learning rate for biases.
        if 'biases' in var.op.name:
            gradient_multipliers[var.op.name] = 2.

        # Use larger learning rate for last layer variables.
        for layer in last_layers:
            if layer in var.op.name and 'biases' in var.op.name:
                gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
                break
            elif layer in var.op.name:
                gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
                break

    return gradient_multipliers